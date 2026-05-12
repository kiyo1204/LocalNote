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
from coordinate_chunker import CoordinateChunker

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

# パラメータの保存
def save_rag_params(db_path, chunk_size, chunk_overlap, k):
    os.makedirs(db_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
    params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k
    }
    with open(os.path.join(db_path, "rag_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

# パラメータのロード
def load_rag_params(db_path):
    param_path = os.path.join(db_path, "rag_params.json")
    if os.path.exists(param_path):
        with open(param_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

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
                            content = result.text_content
                        else:
                            with open(file_path, encoding="utf-8") as f:
                                content = f.read()

                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path}
                        )
                        docs.append(doc)
                    else:
                        try:
                            # 座標ベースチャンキングを使用
                            chunker = CoordinateChunker(eps=15.0, min_chars=3, direction_threshold=0.7)
                            chunks_info = chunker.create_chunks(file_path, min_chunk_size=20)
                            
                            if chunks_info:
                                for chunk in chunks_info:
                                    # Chromaがdictメタデータを受け入れられないため、座標を文字列化
                                    bbox = chunk["bbox"]
                                    bbox_str = f"x:{bbox['x0']:.1f}-{bbox['x1']:.1f},y:{bbox['y0']:.1f}-{bbox['y1']:.1f}"
                                    
                                    doc = Document(
                                        page_content=chunk["text"],
                                        metadata={
                                            "source": file_path,
                                            "page": chunk["page"],
                                            "direction": chunk["direction"],
                                            "bbox": bbox_str,
                                            "char_count": chunk["char_count"]
                                        }
                                    )
                                    if doc.page_content.strip():
                                        docs.append(doc)
                                
                                stats = chunker.get_chunk_stats(chunks_info)
                                st.info(f"PDF読込: {file} ({stats['total_chunks']}チャンク | "
                                        f"横:{stats['horizontal_chunks']} 縦:{stats['vertical_chunks']} 混:{stats['mixed_chunks']})")
                            else:
                                st.warning(f"PDF読込失敗: {file} (チャンク化失敗)")
                        except Exception as pdf_error:
                            st.warning(f"PDF処理エラー: {file} ({pdf_error})")

                    progress.progress(int((i + 1) / len(files) * 100), text)
                except Exception as e:
                    # 変換できない隠しファイルなどはスキップ
                    st.warning(f"変換スキップ: {file} ({e})")
        st.success("ファイルの変換完了")
        return docs

    # 新規データベースの構築
    def build_database(self, dir_path, target_db_path, chunk_size, chunk_overlap, k):
        pages = self._load_documents(dir_path)
        self.k = k
        
        # ドキュメントが読み込まれたか確認
        st.info(f"読込ドキュメント数: {len(pages)}")
        if not pages:
            st.error("読み込むドキュメントがありません")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "# ", "## ", "。", "、", " "]
        )
        chunks = text_splitter.split_documents(pages)
        
        # 確認用
        # st.write(chunks)
        
        # チャンク化の結果を確認
        st.info(f"生成チャンク数: {len(chunks)}")
        if not chunks:
            st.error("チャンク生成に失敗しました")
            return

        if not os.path.exists(target_db_path) or not os.listdir(target_db_path):
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=target_db_path
            )
            st.success(f"データベース構築完了: {len(chunks)}チャンク")

    # 既存データベースへのファイル追加
    def add_to_database(self, new_dir_path, target_db_path, chunk_size, chunk_overlap):
        pages = self._load_documents(new_dir_path)
        
        # ドキュメントが読み込まれたか確認
        st.info(f"読込ドキュメント数: {len(pages)}")
        if not pages:
            st.error("読み込むドキュメントがありません")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "# ", "## ", "。", "、", " "]
        )
        chunks = text_splitter.split_documents(pages)
        
        # チャンク化の結果を確認
        st.info(f"生成チャンク数: {len(chunks)}")
        if not chunks:
            st.error("チャンク生成に失敗しました")
            return

        db = Chroma(persist_directory=target_db_path, embedding_function=self.embeddings)
        db.add_documents(chunks)
        st.success(f"データベース更新完了: {len(chunks)}チャンク追加")
        gc.collect()

    # 指定されたデータベースを検索し、LLMに回答を生成させる
    def ask(self, db_path, query, history_data):
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        k = getattr(self, "k", 6)
        retriever = db.as_retriever(search_kwargs={"k": k})

        chat_history = []
        if history_data and isinstance(history_data, dict):
            for exchange in history_data.values():
                if (
                    isinstance(exchange, dict)
                    and "input" in exchange
                    and "output" in exchange
                ):
                    chat_history.append(
                        HumanMessage(content=exchange["input"].get("content", ""))
                    )
                    chat_history.append(
                        AIMessage(content=exchange["output"].get("content", ""))
                    )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        used_sources = {
            os.path.basename(doc.metadata.get("source", ""))
            for doc in response.get("context", [])
            if isinstance(doc.metadata, dict)
        }

        return {
            "answer": response["answer"],
            "sources": sorted(used_sources)
        }
    
    # 単語帳リストの生成
    def generate_flashcard(self, db_path, num_words):
        db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 20})

        # JSON生成用プロンプト
        create_json_system_prompt = """あなたは優秀な教材作成アシスタントです。

        以下の参考情報から、重要な専門用語を抽出してください。

        【出力形式の厳密な制約】
        - 出力はJSONオブジェクトのみ
        - 説明文・Markdown, latexは禁止
        - 以下の形式を必ず守ること
        - 指定された個数単語を抽出すること
        - 単語は日本語で, JSON以外を出力しない

        【出力形式】
        "terms": [
            {{
            "term": "用語",
            "definition": "意味"
            }}
        ]

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
    
    # 複数ファイル由来のベクトルをDBから削除
    def delete_files_from_database(self, db_path, target_file_paths: list[str]):
        db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        for path in target_file_paths:
            db._collection.delete(
                where={"source": path}
            )

        gc.collect()

    # 資料の文字数推定
    def estimate_total_chars(self, dir_path):
        total_chars = 0
        md = MarkItDown()

        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    if file.lower().endswith(".pdf"):
                        loader = PDFPlumberLoader(file_path)
                        pages = loader.load()
                        total_chars += sum(len(p.page_content) for p in pages)

                    elif file.lower().endswith(".md"):
                        with open(file_path, encoding="utf-8") as f:
                            total_chars += len(f.read())

                    else:
                        result = md.convert(file_path)
                        total_chars += len(result.text_content)

                except Exception:
                    continue

        return total_chars

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

# パラメータの自動決定
def auto_rag_params(total_chars: int):
    # 文字数に基づいてチャンクサイズとK値を決定
    if total_chars < 5_000:
        # 非常に小規模
        chunk_size = 300
        k = 20
    elif total_chars < 20_000:
        # 小規模
        chunk_size = 500
        k = 18
    elif total_chars < 50_000:
        # 中規模
        chunk_size = 800
        k = 15
    elif total_chars < 100_000:
        # 大規模
        chunk_size = 1000
        k = 10
    elif total_chars < 300_000:
        # 超大規模
        chunk_size = 1200
        k = 8
    else:
        # 極大規模
        chunk_size = 1500
        k = 6

    chunk_overlap = int(chunk_size * 0.2)  # 20%のオーバーラップで文脈保持
    return chunk_size, chunk_overlap, k

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

        if st.session_state["db_ready"]:
            st.markdown("---")
            st.header("📂 学習ファイル管理")
            st.subheader("学習ファイルの追加")
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
                        shutil.move(
                            os.path.join(temp_file_dir, file_name),
                            os.path.join(final_file_dir, file_name)
                        )

                    shutil.rmtree(temp_file_dir, ignore_errors=True)

                    st.success("✅ファイルの追加が完了しました")

            st.subheader("🗂 現在の学習ファイル")
            db_name = os.path.basename(st.session_state["current_db"])
            file_dir = os.path.join(PDF_DIR, db_name)

            if os.path.exists(file_dir):
                files = sorted(os.listdir(file_dir))
                selected_files = []
                is_delete_chat = False

                if files:
                    for file in files:
                        checked = st.checkbox(file, key=f"del_{db_name}_{file}")
                        if checked:
                            selected_files.append(file)

                    if selected_files:
                        if len(files) == len(selected_files):
                            is_delete_chat = True
                            st.warning("すべてのファイルが選択されているため，チャットが削除されます")
                        else:
                            st.warning("選択したファイルはデータベースから完全に削除されます")

                        if st.button("❌ 選択したファイルを削除"):
                            with st.spinner("削除中..."):
                                if is_delete_chat:
                                    delete_chat(st.session_state["current_db"], os.path.join(OUTPUT_DIR, db_name))
                                else:
                                    full_paths = [
                                        os.path.join(file_dir, f) for f in selected_files
                                    ]

                                    # DBから削除
                                    st.session_state["rag_engine"].delete_files_from_database(
                                        st.session_state["current_db"],
                                        full_paths
                                    )

                                    # 元ファイル削除
                                    for path in full_paths:
                                        if os.path.exists(path):
                                            os.remove(path)

                                    st.success("✅ 選択した学習ファイルを削除しました")
                                    time.sleep(.5)
                                    st.rerun()
                else:
                    st.info("学習ファイルがありません")
            else:
                st.info("学習フォルダが存在しません")
        
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

                    # RAGパラメータ復元
                    params = load_rag_params(target_db_path)
                    if params:
                        st.session_state["rag_engine"].k = params["k"]

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
                        
                        total_chars = st.session_state["rag_engine"].estimate_total_chars(save_dir)
                        
                        chunk_size, chunk_overlap, k_value = auto_rag_params(total_chars)

                        # DB構築
                        st.session_state["rag_engine"].build_database(
                            save_dir,
                            target_db_path,
                            chunk_size,
                            chunk_overlap,
                            k_value
                        )

                        # パラメータ保存
                        save_rag_params(
                            target_db_path,
                            chunk_size,
                            chunk_overlap,
                            k_value
                        )
                        
                        st.session_state["db_ready"] = True
                        st.session_state["current_db"] = target_db_path
                        st.success("✅データベースの構築が完了しました！")

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
                            result = st.session_state["rag_engine"].ask(
                                st.session_state["current_db"],
                                user_query,
                                st.session_state["history"]
                            )

                            answer = result["answer"]
                            sources = result["sources"]

                            for char in answer:
                                dislpay_char += char
                                placeholder.markdown(dislpay_char)
                                time.sleep(.02)

                            if sources:
                                st.markdown("---")
                                st.markdown("📚 **参考にした資料:**")
                                for src in sources:
                                    st.markdown(f"- {src}")

                            # 履歴データの作成と追加
                            data = {
                                "input": {"role": "user", "content": user_query},
                                "output": {"role": "assistant", "content": answer}
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