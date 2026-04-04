import os
import logging
import streamlit as st
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 初期設定・ログ抑制
DB_DIR = "./db"
PDF_DIR = "./upload_files"
OUTPUT_DIR = "./history"
CONFIG_DIR = "./config/config.json"

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
あなたは優秀なアシスタントです。以下の参考情報のみを用いて、ユーザーの質問に答えてください。
また、まとめる場所は本題が始まった場所からとし、指示がない限り、テスト要項や成績についてはまとめない。
"""

def load_config():
    if os.path.exists(CONFIG_DIR):
        with open(CONFIG_DIR, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"system_prompt": DEFAULT_PROMPT}

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
        self.llm = ChatOllama(model="gemma4", temperature=0)
        self.update_prompt(system_prompt)
        
    def update_prompt(self, new_system_prompt):
        # ファインチューニングの変更
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt + "\n\n参考情報:\n{context}"),
            ("human", "{input}"),
        ])

    def build_database(self, pdf_dir_path, target_db_path):
        # PDFを読み込み、分割してベクトルDBを構築する
        loader = PyPDFDirectoryLoader(pdf_dir_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "、", " "]
        )
        chunks = text_splitter.split_documents(pages)

        if not os.path.exists(target_db_path) or not os.listdir(target_db_path):
            Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=target_db_path)
        else:
            Chroma(persist_directory=target_db_path, embedding_function=self.embeddings)

    def ask(self, db_path, query):
        # 指定されたDBを検索し、LLMに回答を生成させる
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})
        
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response["answer"]


# 履歴マネージャー (JSONファイルの読み書き)
class HistoryManager:
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir

    def load(self, db_name):
        # 特定のDBの履歴をJSONから読み込む
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self, db_name, history_data):
        # 現在の履歴全体をJSONに保存する
        out_dir = os.path.join(self.base_output_dir, db_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "history.json")
        
        with open(out_path, mode="w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)

    def clear(self, db_name):
        # 特定の履歴の削除
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            os.remove(json_path)


# Streamlit UI部分 (フロントエンド)
if __name__ == "__main__":
    # 状態の初期化・設定
    if "config" not in st.session_state:
        st.session_state["config"] = load_config()
    if "rag_engine" not in st.session_state:
        st.session_state["rag_engine"] = RAGEngine(st.session_state["config"]["system_prompt"])
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = HistoryManager(OUTPUT_DIR)

    if "db_ready" not in st.session_state:
        st.session_state["db_ready"] = False
    if "history" not in st.session_state:
        st.session_state["history"] = {}
    if "current_db" not in st.session_state:
        st.session_state["current_db"] = ""
        
    with st.sidebar:
        st.title("LocalNote")

        # メニュー
        app_mode = st.radio("メニュー", ["💭チャット画面", "⚙️設定画面"])
        st.markdown("---")

        st.header("チャットの管理・作成")
        existing_db = [i for i in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, i))]
        db_mode = st.radio("操作の選択", ["既存チャットの読み込み", "新規チャットの作成"])
        
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

                if st.button("作成して実行する"):
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
            st.success(f"チャット接続中: {db_name}")

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
                            ans = st.session_state["rag_engine"].ask(st.session_state["current_db"], user_query)
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
        st.header("設定")

        # ファインチューニング設定
        st.subheader("🤖ファインチューニング設定")
        st.markdown("AIへ行う事前の指示を変更できます。")

        new_prompt = st.text_input(
            "現在のプロンプト",
            value=st.session_state["config"]["system_prompt"],
        )

        if st.button("保存"):
            st.session_state["config"]["system_prompt"] = new_prompt
            save_config(st.session_state["config"])
            st.session_state["rag_engine"].update_prompt(new_prompt)

        st.markdown("---")

        # 履歴の削除
        st.subheader("🗑️ データの削除")
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            st.warning(f"現在接続中のチャット(**{db_name}**)の履歴を削除します。")

            if st.button("削除"):
                st.session_state["history_manager"].clear(db_name)
                st.session_state["history"] = {}
                st.success("履歴の削除に成功しました。")
        else:
            st.info("履歴の削除にはサイドバーから対象のチャットを読み込んでください。")