from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

import os, logging, json, shutil, gc, time, chromadb
import streamlit as st


# 初期設定・ログ抑制
DB_DIR = "./db"
PDF_DIR = "./upload_files"
OUTPUT_DIR = "./history"
CONFIG_DIR = "./config.json"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

# config
DEFAULT_PROMPT = """
あなたは優秀なアシスタントです。以下の参考情報のみを用いて、ユーザーの質問に答えてください。 また、まとめる場所は本題が始まった場所からとし、指示がない限り、テスト要項や成績についてはまとめないでください。
"""

# コンテキスト内学習用jsonの読み込み
def load_config():
    if os.path.exists(CONFIG_DIR):
        with open(CONFIG_DIR, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PROMPT

# コンテキスト内学習用jsonの保存
def save_config(config_data):
    with open(CONFIG_DIR, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


# RAGエンジン (AIの頭脳とデータベース操作)
class RAGEngine:
    def __init__(self, system_prompt):
        # アプリ起動時に一度だけモデルをロードして保持する
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"}
        )
        self.llm = ChatOllama(model=st.session_state["model_name"], temperature=0)

        # 質問再構築用プロンプト
        contextualize_q_system_prompt = """これまでのチャット履歴と、最新のユーザーの質問を受け取ります。
        チャット履歴の文脈がなくても理解できる、独立した質問に再構築してください。
        質問を再構築する必要がない場合は、そのまま返してください。質問に対する回答や説明は絶対にしないでください。
        """
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.update_prompt(system_prompt)
        
    def update_prompt(self, new_system_prompt):
        # コンテキスト内学習の変更
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt + "\n\n参考情報:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # モデル設定
        self.llm = ChatOllama(model=st.session_state["model_name"], temperature=0)

    # PDFを読み込み、分割してベクトルデータベースを構築する
    def build_database(self, pdf_dir_path, target_db_path):
        loader = DirectoryLoader(
            pdf_dir_path,
            glob="**/*.pdf",
            loader_cls=PDFPlumberLoader
        )
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, # チャンクの最大文字数
            chunk_overlap=50, # チャンク間の重複させる文字数
            separators=["\n\n", "\n", "。", "、", " "] # 分割する文字の優先順位
        )
        chunks = text_splitter.split_documents(pages)

        if not os.path.exists(target_db_path) or not os.listdir(target_db_path):
            Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=target_db_path)
        else:
            Chroma(persist_directory=target_db_path, embedding_function=self.embeddings)

    # 指定されたデータベースを検索し、LLMに回答を生成させる
    def ask(self, db_path, query, history_data):
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})

        # 履歴データをLangchainに読み込ませられるように変換
        chat_history = []
        if history_data:
            for key, exchange in history_data.items():
                chat_history.append(HumanMessage(content=exchange["input"]["content"]))
                chat_history.append(AIMessage(content=exchange["output"]["content"]))

        # 履歴を考慮してデータベースを検索するRetriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )

        # 回答生成チェーン
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
            })
        return response["answer"]


# 履歴マネージャー (JSONファイルの読み書き)
class HistoryManager:
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir

    # 特定のデータベースの履歴をjsonから読み込む
    def load(self, db_name):
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # 現在の履歴全体をjsonに保存する
    def save(self, db_name, history_data):
        out_dir = os.path.join(self.base_output_dir, db_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "history.json")
        
        with open(out_path, mode="w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)

    # 特定の履歴の削除
    def clear(self, db_name):
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            os.remove(json_path)

# チャット削除確認ダイアログ
@st.dialog("本当に削除しますか?")
def delete_chat(db_name, history_name):
    st.warning("データは完全に消去されます。本当に削除しますか?")
    col1, col2 = st.columns(2)
    if col1.button("はい"):
        # 状態の初期化
        st.session_state["db_ready"] = False
        st.session_state["current_db"] = ""
        st.session_state["history"] = {}

        # データベースのファイルロック強制解除(エラーが発生したらスルー)
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        # ガベージコレクション(メモリの不要な部分の開放)を実行
        gc.collect()
        time.sleep(1)

        try:
            # 履歴・データベースの削除
            if os.path.exists(db_name):
                shutil.rmtree(db_name, ignore_errors=True)
            if os.path.exists(history_name):
                shutil.rmtree(history_name)

            # PDFの削除
            pdf_dir = os.path.join(PDF_DIR, os.path.basename(db_name))
            if os.path.exists(pdf_dir):
                shutil.rmtree(pdf_dir)

            st.success("削除しました。")
            time.sleep(.5)
            st.rerun()
        except Exception as e:
            st.error(f"削除中にエラーが発生しました: {e}")

    if col2.button("いいえ"):
        st.rerun()

# Streamlit UI部分 (フロントエンド)
if __name__ == "__main__":
    # 状態の初期化・設定
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = "gemma4:e4b"
    if "config" not in st.session_state:
        st.session_state["config"] = load_config()
    if "rag_engine" not in st.session_state:
        st.session_state["rag_engine"] = RAGEngine(st.session_state["config"])
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = HistoryManager(OUTPUT_DIR)
    if "db_ready" not in st.session_state:
        st.session_state["db_ready"] = False
    if "history" not in st.session_state:
        st.session_state["history"] = {}
    if "current_db" not in st.session_state:
        st.session_state["current_db"] = ""
        
    # サイドバー
    with st.sidebar:
        st.title("LocalNote")

        # メニュー
        app_mode = st.radio("メニュー", ["💭チャット画面", "⚙️設定画面"])
        st.markdown("---")

        st.header("チャットの管理・作成")
        existing_db = [i for i in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, i))]
        db_mode = st.radio("操作の選択", ["既存チャットの読み込み", "新規チャットの作成"], index=0)
        
        if db_mode == "既存チャットの読み込み":
            if existing_db:
                selected_db = st.selectbox("使用チャットの選択", existing_db)
                target_db_path = os.path.join(DB_DIR, selected_db)

                if st.button("開始"):
                    st.session_state["db_ready"] = True
                    st.session_state["current_db"] = target_db_path
                    
                    st.session_state["history"] = st.session_state["history_manager"].load(selected_db)
                    st.success(f"✅[{selected_db}]を読み込みました!")
            else:
                st.warning("既存のチャットがありません。'新規チャットの作成'から作成してください。")
        else:
            new_db_name = st.text_input("新しいチャット名を入力")
            if new_db_name:
                target_db_path = os.path.join(DB_DIR, new_db_name)

            files = st.file_uploader("まとめたいファイルのアップロード", accept_multiple_files=True, type="pdf")
            
            if st.button("作成して実行する"):
                if files and new_db_name:
                    progress_bar = st.progress(0, text="ファイルを保存しています")
                    save_dir = os.path.join(PDF_DIR, new_db_name)
                    os.makedirs(save_dir, exist_ok=True)
                
                    for i, file in enumerate(files):
                        file_path = os.path.join(save_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        progress_bar.progress(int((i + 1) / len(files) * 100), text="ファイルを保存しています")
                    progress_bar.empty()

                
                    st.session_state["history"] = {}
                    
                    progress_bar = st.progress(0, text="データベースの構築中")
                    st.session_state["rag_engine"].build_database(save_dir, target_db_path)
                    progress_bar.progress(100, text="構築完了")
                    
                    st.session_state["db_ready"] = True
                    st.session_state["current_db"] = target_db_path
                    st.success("✅データベースの構築が完了しました！")
                    progress_bar.empty()

    # チャット画面
    if app_mode == "💭チャット画面":
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            st.success(f"チャット接続中: [**{db_name}**], モデル: [**{st.session_state['model_name']}**]")

            # 履歴を表示
            if st.session_state["history"]:
                for key, exchange in st.session_state["history"].items():
                    with st.chat_message(exchange["input"]["role"]):
                        st.markdown(exchange["input"]["content"])
                    with st.chat_message(exchange["output"]["role"]):
                        st.markdown(exchange["output"]["content"])
                        
            user_query = st.chat_input("質問を入力してください...")
                
            if user_query:
                with st.chat_message("user"):
                    st.write(user_query)
                    
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        try:
                            ans = st.session_state["rag_engine"].ask(
                                st.session_state["current_db"],
                                user_query,
                                st.session_state["history"]
                                )
                            st.markdown(ans)

                            # 履歴データの作成と追加
                            data = {
                                "input": {"role": "user", "content": user_query},
                                "output": {"role": "assistant", "content": ans}
                            }
                            st.session_state["history"]["question_"+str(len(st.session_state["history"]))] = data

                            # HistoryManagerを使って履歴をセーブ
                            st.session_state["history_manager"].save(db_name, st.session_state["history"])
                                    
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
        else:
            st.info("チャットを開始するにはサイドバーからチャットを選択してください。")
    else:
        st.title("設定")
        # モデル設定
        st.subheader("🤖モデル設定")
        st.markdown("使用するモデルを設定できます。環境に合わせて設定してください。<br>また、モデルの使用にはモデルのダウンロードが必要です。", unsafe_allow_html=True)
        model_name = st.selectbox("モデル設定", ["gemma4:e4b", "gemma4:e2b", "その他"])
        if model_name == "その他":
            model_name = st.text_input("モデル名を入力してください。")

        # コンテキスト内学習設定
        st.subheader("🤖コンテキスト内学習グ設定")
        st.markdown("AIへ行う事前の指示を変更できます。")
        new_prompt = st.text_input(
            "現在のプロンプト",
            value=st.session_state["config"],
        )

        if st.button("保存"):
            st.session_state["config"] = new_prompt
            save_config(new_prompt)
            st.session_state["model_name"] = model_name
            st.session_state["rag_engine"].update_prompt(new_prompt)
        st.markdown("---")

        # 履歴の削除
        st.subheader("🗑️ 履歴の削除")
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            st.warning(f"現在接続中のチャット(**{db_name}**)の履歴を削除します。")

            if st.button("履歴削除"):
                st.session_state["history_manager"].clear(db_name)
                st.session_state["history"] = {}
                st.success("履歴の削除に成功しました。")
        else:
            st.info("履歴の削除にはサイドバーから対象のチャットを読み込んでください。")
            
        st.markdown("---")
        st.subheader("🚨チャットの削除(履歴も)")
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            st.warning(f"現在接続中のチャット(**{db_name}**)のデータ全てを削除します。")

            if st.button("削除"):
                delete_chat(st.session_state["current_db"], os.path.join(OUTPUT_DIR, db_name))