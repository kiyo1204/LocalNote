"""
RAG Engine for LocalNote.
Handles document loading, vector database operations, retrieval, and LLM interaction.
"""

import os, logging, json, gc, time, re
from typing import Any, List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from markitdown import MarkItDown

from src.coordinate_chunker import CoordinateChunker
from src.reranker import RerankerFactory
from src.config_manager import DEFAULT_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from src.utils import _extract_json

logger = logging.getLogger(__name__)


class RAGEngine:
    """AI brain and database operations for RAG."""

    def __init__(self, system_prompt, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"}
        )
        self.llm = ChatOllama(
            model=st.session_state["model_name"],
            temperature=temperature,
            num_predict=max_tokens
        )
        self.first_model_name = st.session_state["model_name"]

        # モデルキャッシュ
        self.db_cache = {}

        # Retrieval設定
        self.retriever_type = "similarity"
        self.enable_reranking = False
        self.rerank_model = "noop"
        self.mmr_lambda_mult = 0.5
        self.fair_search = True
        self.max_history_pairs = 6
        self.smart_forgetting = True  # 重要情報を保持し、不要情報を忘却

        self._sources_cache: dict = {}

        self.update_prompt(system_prompt)

        # 質問再構築用プロンプト
        contextualize_q_system_prompt = """あなたはチャット履歴を考慮して質問を言い換えるアシスタントです。

## 役割
ユーザーの最新の質問を、過去の会話がなくても単独で理解できる形に再構築してください。

## ルール
1. 質問が既に独立している（履歴不要）場合は、そのまま返してください。
2. 「それ」「これ」「あれ」「その」「同じ」「上記」などの指示語がある場合は、具体的な名前に置き換えてください。
3. 省略された文脈（例：「次は？」→「次の章/項目について教えて」）を補完してください。
4. 質問の意図（例：要約、説明、比較、列挙）を保持したまま言い換えてください。
5. 絶対に質問に回答したり説明を追加したりしないでください。
6. 再構築した質問のみを出力し、前置き・補足は一切付け加えないでください。
"""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def get_db(self, db_path):
        if db_path not in self.db_cache:
            self.db_cache[db_path] = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
        return self.db_cache[db_path]

    def update_prompt(self, new_system_prompt):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt + "\n\n---\n【参考情報】\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def update_llm(self, model_name=None, temperature=None, max_tokens=None):
        self.llm = ChatOllama(
            model=model_name or st.session_state["model_name"],
            temperature=temperature if temperature is not None else self.llm.temperature,
            num_predict=max_tokens if max_tokens is not None else getattr(self.llm, "num_predict", DEFAULT_MAX_TOKENS)
        )

    def _get_unique_sources(self, db, db_path: str) -> list:
        if db_path in self._sources_cache:
            return self._sources_cache[db_path]
        try:
            result = db.get(include=["metadatas"])
            metadatas = result.get("metadatas", []) or []
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            sources_list = sorted(sources)
            self._sources_cache[db_path] = sources_list
            return sources_list
        except Exception as e:
            logger.warning(f"Failed to get unique sources: {e}")
            return []

    def _create_fair_retriever(self, db, db_path: str, k: int):
        sources = self._get_unique_sources(db, db_path)
        if len(sources) <= 1:
            logger.info("Fair search: only one source file, using standard retriever")
            return db.as_retriever(search_kwargs={"k": k})

        per_file_k = max(1, k // len(sources))
        logger.info(f"Fair search: {len(sources)} files, per_file_k={per_file_k}, k={k}")

        class FairRetriever(BaseRetriever):
            db: Any = None
            sources: List[str] = []
            per_file_k: int = 1
            k: int = 1

            def _get_relevant_documents(self, query):
                file_results = []
                for source in self.sources:
                    try:
                        docs = self.db.similarity_search(query, k=self.per_file_k, filter={"source": source})
                        if docs:
                            file_results.append(docs)
                    except Exception as e:
                        logger.warning(f"Fair search error for {source}: {e}")
                if not file_results:
                    return []
                result = []
                max_len = max(len(docs) for docs in file_results)
                for i in range(max_len):
                    for docs in file_results:
                        if i < len(docs):
                            result.append(docs[i])
                            if len(result) >= self.k:
                                return result
                return result

        return FairRetriever(db=db, sources=sources, per_file_k=per_file_k, k=k)

    def _create_enhanced_retriever(self, db_path: str, k: int):
        db = self.get_db(db_path)

        if self.fair_search:
            base_retriever = self._create_fair_retriever(db, db_path, k)
        else:
            if self.retriever_type == "mmr":
                base_retriever = db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": k, "lambda_mult": self.mmr_lambda_mult}
                )
            elif self.retriever_type == "hybrid":
                base_retriever = db.as_retriever(search_kwargs={"k": k * 2})
            else:
                base_retriever = db.as_retriever(search_kwargs={"k": k})

        if self.enable_reranking:
            reranker = RerankerFactory.get_reranker(self.rerank_model)

            class RerankingRetriever(BaseRetriever):
                base_retriever: Any = None
                reranker: Any = None
                k: int = 1

                def _get_relevant_documents(self, query):
                    docs = self.base_retriever._get_relevant_documents(query)
                    reranked = self.reranker.rerank(query, docs)
                    return [doc for doc, score in reranked[:self.k]]

            return RerankingRetriever(base_retriever=base_retriever, reranker=reranker, k=k)

        return base_retriever

    def _chunk_documents(self, pages, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "# ", "## ", "。", "、", " "]
        )
        chunks = []
        for page in pages:
            if "bbox" in page.metadata:
                chunks.append(page)
            else:
                chunks.extend(text_splitter.split_documents([page]))
        return chunks

    def _load_documents(self, dir_path):
        md = MarkItDown()
        docs = []
        self._coord_chunker = CoordinateChunker(eps=15.0, min_chars=3, direction_threshold=0.7)
        text = "ファイルの変換中..."
        progress = st.progress(0, text)

        for root, _, files in os.walk(dir_path):
            for i, file in enumerate(files):
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith(".md"):
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                        doc = Document(page_content=content, metadata={"source": file_path})
                        docs.append(doc)
                    elif not file.lower().endswith(".pdf"):
                        result = md.convert(file_path)
                        content = result.text_content
                        doc = Document(page_content=content, metadata={"source": file_path})
                        docs.append(doc)
                    else:
                        try:
                            chunks_info = self._coord_chunker.create_chunks(file_path, min_chunk_size=20)
                            if chunks_info:
                                for chunk in chunks_info:
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
                                stats = self._coord_chunker.get_chunk_stats(chunks_info)
                                st.info(f"PDF読込: {file} ({stats['total_chunks']}チャンク | "
                                        f"横:{stats['horizontal_chunks']} 縦:{stats['vertical_chunks']} 混:{stats['mixed_chunks']})")
                            else:
                                st.warning(f"PDF読込失敗: {file} (チャンク化失敗)")
                        except Exception as pdf_error:
                            st.warning(f"PDF処理エラー: {file} ({pdf_error})")

                    progress.progress(int((i + 1) / len(files) * 100), text)
                except Exception as e:
                    st.warning(f"変換スキップ: {file} ({e})")
        st.success("ファイルの変換完了")
        return docs

    def build_database(self, dir_path, target_db_path, chunk_size, chunk_overlap, k):
        pages = self._load_documents(dir_path)
        self.k = k

        st.info(f"読込ドキュメント数: {len(pages)}")
        if not pages:
            st.error("読み込むドキュメントがありません")
            return

        chunks = self._chunk_documents(pages, chunk_size, chunk_overlap)
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

    def add_to_database(self, new_dir_path, target_db_path, chunk_size, chunk_overlap):
        pages = self._load_documents(new_dir_path)
        st.info(f"読込ドキュメント数: {len(pages)}")
        if not pages:
            st.error("読み込むドキュメントがありません")
            return

        chunks = self._chunk_documents(pages, chunk_size, chunk_overlap)
        st.info(f"生成チャンク数: {len(chunks)}")
        if not chunks:
            st.error("チャンク生成に失敗しました")
            return

        db = self.get_db(target_db_path)
        db.add_documents(chunks)
        st.success(f"データベース更新完了: {len(chunks)}チャンク追加")
        self.db_cache.pop(target_db_path, None)
        self._sources_cache.pop(target_db_path, None)

    # ------------------------------------------------------------------ 
    # Smart Forgetting: 重要会話を保持し、不要会話を忘却
    # ------------------------------------------------------------------ 
    @staticmethod
    def _has_korean(text: str) -> bool:
        """Check if text contains any Korean (Hangul) characters."""
        korean_ranges = [
            (0xAC00, 0xD7AF),  # Hangul Syllables
            (0x1100, 0x11FF),  # Hangul Jamo
            (0x3130, 0x318F),  # Hangul Compatibility Jamo
            (0xA960, 0xA97F),  # Hangul Extended-A
            (0xD7B0, 0xD7FF),  # Hangul Extended-B
        ]
        for char in text:
            code = ord(char)
            for start, end in korean_ranges:
                if start <= code <= end:
                    return True
        return False

    @staticmethod
    def _score_conversation_pair(user_msg: str, ai_msg: str) -> float:
        """
        会話ペアの重要度をスコアリング（0.0 = 忘却対象, 1.0 = 最大重要）
        """
        score = 0.5  # 基準値
        user_len = len(user_msg)
        ai_len = len(ai_msg)

        # --- 重要度が高いシグナル ---
        # 長い質問 → 深い議論をしている
        if user_len > 150:
            score += 0.25
        elif user_len > 80:
            score += 0.15

        # 内容のある質問表現
        if re.search(r'とは|について|教えて|説明|違い|関係|理由|なぜ|どうやって|方法', user_msg):
            score += 0.2

        # 専門用語・固有名詞を含む
        if re.search(r'[A-Z][a-z]+モデル|アルゴリズム|手法|理論|定理|関数|クラス|API|DB|SQL|PDF|RAG|LLM', user_msg):
            score += 0.15

        # 長い回答 → 重要な情報を含む可能性が高い
        if ai_len > 300:
            score += 0.15

        # 命令・要求を含む
        if re.search(r'要約|まとめて|翻訳|比較|一覧|リスト|列挙|箇条書き', user_msg):
            score += 0.1

        # --- 重要度が低いシグナル ---
        # 極端に短い発言
        if user_len < 8:
            score -= 0.25
        elif user_len < 20:
            score -= 0.1

        # 挨拶・感謝・相槌
        if re.search(r'^(こんにちは|こんばんは|おはよう|ありがとう|わかりました|了解|なるほど|すごい|はい|いいえ|OK|ok)$', user_msg.strip()):
            score -= 0.3

        # 短い確認
        if re.search(r'^(そう|なるほど|なるほどですね|あー|えっと|うーん)$', user_msg.strip()):
            score -= 0.3

        # 単なる繰り返し・相槌のAI応答
        if ai_len < 15 and user_len < 15:
            score -= 0.2

        return max(0.05, min(1.0, score))

    def _select_history_pairs(self, pairs: list, max_pairs: int) -> list:
        """
        重要度ベースで会話履歴ペアを選択。
        - 直近の会話は常に保持（最低2ペア）
        - 残りの枠は過去も含めて重要度上位で埋める
        - 重要でない会話（挨拶・短い確認など）は優先的に忘却
        """
        n = len(pairs)
        if n <= max_pairs or not self.smart_forgetting:
            # 枠に収まっているか、Smart Forgettingが無効 → 従来通り末尾N件
            return pairs[-max_pairs:]

        # 最低限の直近会話をリザーブ（会話の流れを維持）
        recent_keep = min(2, max_pairs - 1)

        # 古い会話プールと直近会話に分離
        old_pairs = pairs[:-recent_keep]
        recent_pairs = pairs[-recent_keep:]

        # 残りの枠
        remaining_slots = max_pairs - recent_keep

        # 古い会話を重要度でスコアリング（インデックスも保持）
        scored = []
        for i, (human, ai) in enumerate(old_pairs):
            s = self._score_conversation_pair(human.content, ai.content)
            scored.append((s, i, human, ai))

        # 重要度上位を抽出（スコア降順 → 古い順）
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = scored[:remaining_slots]

        # 元の時系列順にソートしてから直近会話と結合
        selected.sort(key=lambda x: x[1])  # インデックスでソート
        selected_old = [(h, a) for _, _, h, a in selected]

        return selected_old + recent_pairs

    def ask(self, db_path, query, history_data):
        k = getattr(self, "k", 6)
        retriever = self._create_enhanced_retriever(db_path, k)

        chat_history = []
        if history_data and isinstance(history_data, dict):
            # 履歴のペアを取得し、Smart Forgettingで選択
            pairs = []
            for exchange in history_data.values():
                if isinstance(exchange, dict) and "input" in exchange and "output" in exchange:
                    pairs.append((
                        HumanMessage(content=exchange["input"].get("content", "")),
                        AIMessage(content=exchange["output"].get("content", ""))
                    ))
                # Smart Forgetting: 重要情報を保持し、不要情報を忘却
            pairs = self._select_history_pairs(pairs, self.max_history_pairs)
            for human_msg, ai_msg in pairs:
                chat_history.append(human_msg)
                chat_history.append(ai_msg)

        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, self.contextualize_q_prompt)

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="【{source}】\n{page_content}"
        )
        question_answer_chain = create_stuff_documents_chain(
            self.llm, self.prompt,
            document_prompt=document_prompt,
            document_separator="\n\n---\n"
        )
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query, "chat_history": chat_history})

        used_sources = {
            os.path.basename(doc.metadata.get("source", ""))
            for doc in response.get("context", [])
            if isinstance(doc.metadata, dict)
        }

        return {"answer": response["answer"], "sources": sorted(used_sources)}

    def generate_flashcard(self, db_path, num_words):
        retriever = self._create_enhanced_retriever(db_path, 20)

        create_json_system_prompt = """あなたは教材から重要な専門用語とその定義を抽出するアシスタントです。

        【最重要ルール：絶対に日本語のみで出力すること】
        - 用語・定義ともに**必ず日本語のみ**で記述してください
        - **韓国語・中国語・英語など、日本語以外の言語での出力は一切禁止**
        - 参考情報が日本語以外を含んでいても、出力はすべて日本語に翻訳してから出力してください

        【出力ルール】
        - 以下の参考情報から学習者にとって重要な専門用語を抽出してください
        - 定義は学習者が理解しやすい簡潔な日本語で記述してください
        - 同じ概念を指す別の用語（類義語）がある場合は、主要な用語1つに絞ってください
        - 単なる固有名詞（ファイル名、著者名など）は用語として抽出しないでください

        【出力形式の厳格な制約】
        - 出力は以下のJSONオブジェクトのみ（Markdownのコードフェンスは使用禁止）
        - 説明文・Markdown・議論コード・注釈は一切出力禁止
        - 指定された個数ぴったり抽出すること（足りない場合は抽出できた分だけ）
        - 用語と定義は**必ず日本語**で記述（韓国語・中国語・英語は絶対に禁止）

        {{
            "terms": [
                {{
                    "term": "用語",
                    "definition": "意味"
                }}
            ]
        }}

        - "terms" 配列は必須
        - "term" と "definition" 以外のキーは禁止

        参考情報:
        {context}
        """


        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="【{source}】\n{page_content}"
        )
        create_json_prompt = ChatPromptTemplate.from_messages([
            ("system", create_json_system_prompt),
            ("human", "重要な専門用語をJSON形式で{num_words}個抽出してください。この時に制約を厳守してください"),
        ])

        json_chain = create_stuff_documents_chain(
            self.llm, create_json_prompt,
            document_prompt=document_prompt,
            document_separator="\n\n---\n"
        )
        json_rag_chain = create_retrieval_chain(retriever, json_chain)

        # 最大リトライ回数
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                human_input = f"重要な専門用語をJSON形式で{num_words}個出力してください。この時に制約を厳守してください"
                if attempt > 0:
                    human_input += "\n\n【注意】前回の出力に日本語以外の言語（韓国語など）が含まれていました。必ず日本語のみで出力してください。"

                json_response = json_rag_chain.invoke({
                    "input": human_input,
                    "num_words": num_words
                })

                raw_json = json_response.get("answer", "").strip()
                raw_json = re.sub(r"^```(?:json)?\s*\n?", "", raw_json, flags=re.IGNORECASE)
                raw_json = re.sub(r"\n?```\s*$", "", raw_json)
                raw_json = raw_json.strip()

                parsed = _extract_json(raw_json)
                if parsed is None:
                    last_error = ValueError(f"JSONの生成に失敗しました。\nモデル出力:\n{raw_json}")
                    continue

                if isinstance(parsed, dict):
                    terms = parsed.get("terms", [])
                elif isinstance(parsed, list):
                    terms = parsed
                else:
                    last_error = ValueError(f"予期しないJSON形式です: {type(parsed).__name__}")
                    continue

                if not terms:
                    last_error = ValueError("用語が抽出できませんでした（空の配列）")
                    continue

                # 韓国語チェック
                has_korean = False
                for item in terms:
                    term = item.get("term", "").strip()
                    definition = item.get("definition", "").strip()
                    if self._has_korean(term) or self._has_korean(definition):
                        has_korean = True
                        logger.warning(f"韓国語を検出: term='{term}', definition='{definition}'")
                        break

                if has_korean:
                    last_error = ValueError("出力に韓国語が含まれています。リトライします。")
                    logger.warning(f"Flashcard generation attempt {attempt + 1}: Korean detected, retrying...")
                    continue

                # 正常終了: CSV生成
                csv_lines = []
                for item in terms:
                    term = item.get("term", "").strip()
                    definition = item.get("definition", "").strip()
                    if not term or not definition:
                        continue
                    term = term.replace("\n", " ").replace(",", "、")
                    definition = definition.replace("\n", " ").replace(",", "、")
                    csv_lines.append(f"{term},{definition}")

                if attempt > 0:
                    logger.info(f"Flashcard generation succeeded after {attempt + 1} attempts")

                return "\n".join(csv_lines)

            except ValueError as e:
                last_error = e
                continue

        # 全リトライ失敗
        raise last_error or ValueError("単語帳の生成に失敗しました（不明なエラー）")

    def dispose_db(self, db_path):
        db = self.db_cache.pop(db_path, None)
        if db is not None:
            try:
                db.delete_collection()
            except Exception:
                pass
            finally:
                del db
        self._sources_cache.pop(db_path, None)
        gc.collect()
        time.sleep(0.1)

    def delete_files_from_database(self, db_path, target_file_paths: list[str]):
        db = self.get_db(db_path)
        for path in target_file_paths:
            try:
                db._collection.delete(where={"source": path})
            except Exception:
                db.delete(filter={"source": path})
        self.db_cache.pop(db_path, None)
        self._sources_cache.pop(db_path, None)
        gc.collect()

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
