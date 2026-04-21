from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from markitdown import MarkItDown

import os, logging, json, shutil, gc, time, chromadb, re
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
def load_config(mode="config"):
    if os.path.exists(CONFIG_DIR):
        with open(CONFIG_DIR, "r", encoding="utf-8") as f:
            data = json.load(f)
            if mode == "config":
                return data["config"]
            else:
                return data["model"]
    if mode == "config":
        return DEFAULT_PROMPT
    else:
        return "gemma4:e4b"

# コンテキスト内学習用jsonの保存
def save_config(config_data, model_name):
    data = {
        "config": config_data,
        "model": model_name
        }
    with open(CONFIG_DIR, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# RAGエンジン (AIの頭脳とデータベース操作)
class RAGEngine:
    def __init__(self, system_prompt):
        # アプリ起動時に一度だけモデルをロードして保持する
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base", # スペックによってはbase
            model_kwargs={"device": "cpu"}
        )
        self.llm = ChatOllama(
            model=st.session_state["model_name"],
            temperature=0,
            num_predict=4096
            )

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

    # Markdownに変換
    def _load_documents(self, dir_path):
        md = MarkItDown()
        docs = []
        text = "ファイルの変換中..."
        progress = st.progress(0, text)
        
        # フォルダ内のファイルを順番に処理
        for root, _, files in os.walk(dir_path):
            for i, file in enumerate(files):
                file_path = os.path.join(root, file)
                try:
                    if file[-3:] != "pdf":
                        if file[-2:] != "md":
                            # MarkItDownでファイルをマークダウンテキストに変換
                            result = md.convert(file_path)

                        doc = Document(
                            page_content=result.text_content, 
                            metadata={"source": file_path}
                        )
                        docs.append(doc)
                    else:
                        loader = PDFPlumberLoader(file_path)
                        doc = loader.load()
                        docs.extend(doc)
                    progress.progress((i//len(files)+1)*100, text)
                except Exception as e:
                    # 変換できない隠しファイルなどはスキップ
                    print(f"変換スキップ: {file_name} ({e})")
        return docs

    # 新規データベースの構築
    def build_database(self, dir_path, target_db_path):
        pages = self._load_documents(dir_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "# ", "## ", "。", "、", ". ", ", ", " "]
        )
        chunks = text_splitter.split_documents(pages)

        if not os.path.exists(target_db_path) or not os.listdir(target_db_path):
            Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=target_db_path)
        else:
            Chroma(persist_directory=target_db_path, embedding_function=self.embeddings)

    # 既存データベースへのファイル追加
    def add_to_database(self, new_dir_path, target_db_path):
        pages = self._load_documents(new_dir_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "# ", "## ", "。", "、", " "]
        )
        chunks = text_splitter.split_documents(pages)

        db = Chroma(persist_directory=target_db_path, embedding_function=self.embeddings)
        db.add_documents(chunks)
        gc.collect()

    # 指定されたデータベースを検索し、LLMに回答を生成させる
    def ask(self, db_path, query, history_data):
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 20})

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
    
    
    def generate_flashcard(self, db_path, num_words):
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 20})

        # JSON生成用プロンプト
        create_json_system_prompt = """あなたは優秀な教材作成アシスタントです。

        以下の参考情報から、重要な専門用語を抽出してください。

        【出力形式の厳密な制約】
        - 出力はJSONオブジェクトのみ
        - 説明文・Markdownは禁止
        - 以下の形式を必ず守ること
        - 指定された個数単語を抽出すること

        {{
        "terms": [
            {{
            "term": "用語",
            "definition": "意味"
            }}
        ]
        }}

        - 配列は必須
        - term / definition 以外のキーは禁止

        参考情報:
        {context}
        """


        create_json_prompt = ChatPromptTemplate.from_messages([
            ("system", create_json_system_prompt),
            ("human", "重要な専門用語をJSON形式で{num_words}個抽出してください。この時に制約を厳守してください"),
        ])

        # JSON生成（RAG）
        json_chain = create_stuff_documents_chain(self.llm, create_json_prompt)
        json_rag_chain = create_retrieval_chain(retriever, json_chain)

        json_response = json_rag_chain.invoke({
            "input": "重要な専門用語をJSON形式で{num_words}個出力してください。この時に制約を厳守してください",
            "num_words": num_words
        })
        
        raw_json = json_response.get("answer", "").strip()
        
        # コードフェンス除去
        raw_json = re.sub(r"^\s*```(?:json)?\s*", "", raw_json, flags=re.IGNORECASE)
        raw_json = re.sub(r"\s*```\s*$", "", raw_json).strip()

        # 余分な末尾 } を削除
        while raw_json.endswith("}") and raw_json.count("{") < raw_json.count("}"):
            raw_json = raw_json[:-1].rstrip()

        # 余分な末尾 ] を削除
        while raw_json.endswith("]") and raw_json.count("[") < raw_json.count("]"):
            raw_json = raw_json[:-1].rstrip()

        # 足りない } を追加
        while raw_json.count("{") > raw_json.count("}"):
            raw_json += "}"

        # 足りない ] を追加
        while raw_json.count("[") > raw_json.count("]"):
            raw_json += "]"        

        try:
            parsed = json.loads(raw_json)

        except json.JSONDecodeError as e:
            raise ValueError(f"JSONのパースに失敗しました:\n{e}\n\n{raw_json}")

        if isinstance(parsed, dict):
            terms = parsed.get("terms", [])
        elif isinstance(parsed, list):
            terms = parsed
        else:
            raise ValueError("JSONの形式が不正です")

        # CSV生成
        csv_lines = []

        for item in terms:
            term = item.get("term", "").strip()
            definition = item.get("definition", "").strip()

            if not term or not definition:
                continue

            # CSV破壊対策
            term = term.replace("\n", " ").replace(",", "、")
            definition = definition.replace("\n", " ").replace(",", "、")

            csv_lines.append(f"{term},{definition}")

        return "\n".join(csv_lines)



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
                shutil.rmtree(history_name, ignore_errors=True)

            # PDFの削除
            pdf_dir = os.path.join(PDF_DIR, os.path.basename(db_name))
            if os.path.exists(pdf_dir):
                shutil.rmtree(pdf_dir, ignore_errors=True)

            st.success("削除しました。")
            time.sleep(.5)
            st.rerun()
        except Exception as e:
            st.error(f"削除中にエラーが発生しました: {e}")

    if col2.button("いいえ"):
        st.rerun()

# Streamlit UI部分 (フロントエンド)
if __name__ == "__main__":
    st.set_page_config(page_title="LocalNote", page_icon=":shark:")
    # 状態の初期化・設定
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = load_config(mode="model")
    if "config" not in st.session_state:
        st.session_state["config"] = load_config(mode="config")
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
    if "flashcard_csv" not in st.session_state:
        st.session_state["flashcard_csv"] = None
        
    # サイドバー
    with st.sidebar:
        st.title("📒LocalNote")

        # メニュー
        app_mode = st.radio("メニュー", ["💭チャット画面", "⚙️設定画面"])
        st.markdown("---")

        st.header("📖チャットの管理・作成")
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
                    st.rerun()
                    st.success(f"✅[{selected_db}]を読み込みました!")
            else:
                st.warning("既存のチャットがありません。'新規チャットの作成'から作成してください。")
        else:
            new_db_name = st.text_input("新しいチャット名を入力")
            can_exe = True
            if new_db_name:
                target_db_path = os.path.join(DB_DIR, new_db_name)
                if os.path.isdir(target_db_path):
                    st.warning("このチャット名は使用できません。他のチャット名にしてください。")
                    can_exe = True # 実行不可能かどうか
                else:
                    can_exe = False

            files = st.file_uploader("まとめたいファイルのアップロード", accept_multiple_files=True, type=["pdf", "md", "docx", "xlsx", "pptx"], disabled=can_exe)
            
            if st.button("作成して実行する"):
                if files and new_db_name:
                    with st.spinner("実行中..."):
                        save_dir = os.path.join(PDF_DIR, new_db_name)
                        os.makedirs(save_dir, exist_ok=True)
                    
                        for i, file in enumerate(files):
                            file_path = os.path.join(save_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                    
                        st.session_state["history"] = {}
                        
                        st.session_state["rag_engine"].build_database(save_dir, target_db_path)
                        
                        st.session_state["db_ready"] = True
                        st.session_state["current_db"] = target_db_path
                        st.success("✅データベースの構築が完了しました！")
        st.markdown("---")
        st.header("学習データの追加")
        if st.session_state["db_ready"]:
            db = st.session_state["current_db"]
            files = st.file_uploader("追加したいデータをアップロード", accept_multiple_files=True)
            if st.button("追加の実行"):
                temp_file_dir = os.path.join(PDF_DIR, "temp")
                os.makedirs(temp_file_dir, exist_ok=True)

                for i, file in enumerate(files):
                    with open(os.path.join(temp_file_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())

                with st.spinner("追加資料を解析中..."):
                    st.session_state["rag_engine"].add_to_database(temp_file_dir, db)

                    final_file_dir = os.path.join(PDF_DIR, os.path.basename(st.session_state["current_db"]))
                    for file_name in os.listdir(temp_file_dir):
                        shutil.move(os.path.join(temp_file_dir, file.name), os.path.join(final_file_dir, file_name))

                        shutil.rmtree(temp_file_dir)
                        st.success("✅ファイルの追加が完了しました")
        else:
            st.info("追加を行うにはまずチャットの選択・作成を行ってください")
        
        st.markdown("---")
        
        st.subheader("📝 単語帳データの出力")
        num_words = st.slider("単語の数", min_value=5, max_value=40)
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            
            if st.button("単語帳データを生成"):
                with st.spinner(f"AIが用語を抽出中...\n(数分かかることがあります)"):
                    try:
                        # 抽出の実行
                        st.session_state["flashcard_csv"] = st.session_state["rag_engine"].generate_flashcard(
                            st.session_state["current_db"],
                            num_words
                        )
                        
                        st.success(f"✅ ファイルを作成しました！")
                    except Exception as e:
                        st.error(f"生成中にエラーが発生しました: {e}")
            
            # 生成されたデータがあればダウンロード可
            if st.session_state["flashcard_csv"] is not None:
                csv_bytes = st.session_state["flashcard_csv"].encode("utf-8-sig")
                st.download_button(
                    label="📥 CSVファイルをダウンロード",
                    data=csv_bytes,
                    file_name=f"flashcards_{db_name}.csv",
                    mime="text/csv"
                )
                st.session_state["flashcard_csv"] = None

        else:
            st.info("チャットを読み込むと生成できるようになります。")


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
                        dislpay_char = ""
                        placeholder = st.empty()
                        try:
                            ans = st.session_state["rag_engine"].ask(
                                st.session_state["current_db"],
                                user_query,
                                st.session_state["history"]
                                )
                            for char in ans:
                                dislpay_char += char
                                placeholder.markdown(dislpay_char)
                                time.sleep(.02)

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
        st.subheader("🤖コンテキスト内学習設定")
        st.markdown("AIへ行う事前の指示を変更できます。")
        new_prompt = st.text_input(
            "現在のプロンプト",
            value=st.session_state["config"],
        )

        if st.button("保存"):
            st.session_state["config"] = new_prompt
            st.session_state["model_name"] = model_name
            save_config(new_prompt, model_name)
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