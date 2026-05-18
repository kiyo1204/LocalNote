"""
LocalNote - Streamlit UI Application
Main UI including chat, settings, and tips/help viewer.
"""

import os, logging, json, shutil, gc, time, re
import streamlit as st

from src.config_manager import (
    load_config, save_config, save_rag_params, load_rag_params,
    DEFAULT_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, PROMPT_PRESETS
)
from src.rag_engine import RAGEngine
from src.history_manager import HistoryManager
from src.utils import auto_rag_params, _extract_json, delete_chat_files


# ディレクトリ定数
DB_DIR = "./db"
PDF_DIR = "./upload_files"
OUTPUT_DIR = "./history"
DOCS_DIR = "./docs"

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


# ------------------------------------------------
# 削除確認ダイアログ
# ------------------------------------------------
@st.dialog("本当に削除しますか?")
def delete_chat(db_name, history_name):
    st.warning("データは完全に消去されます。本当に削除しますか?")
    col1, col2 = st.columns(2)
    if col1.button("はい"):
        pdf_dir = os.path.join(PDF_DIR, os.path.basename(db_name))
        delete_chat_files(db_name, history_name, pdf_dir)
    if col2.button("いいえ"):
        st.rerun()


# ------------------------------------------------
# Tips表示用のヘルパー
# ------------------------------------------------
def render_tips_page():
    st.title("📖 Tips / ヘルプ")
    st.markdown("このアプリの機能や設定方法について解説したドキュメントを読めます。")

    # docs/ から .md ファイルをスキャン
    md_files = []
    if os.path.exists(DOCS_DIR):
        for f in sorted(os.listdir(DOCS_DIR)):
            if f.lower().endswith(".md"):
                md_files.append(f)

    if not md_files:
        st.info("Tipsドキュメントが見つかりません。`docs/` ディレクトリに .md ファイルを配置してください。")
        return

    # ファイル選択 + ナビゲーション
    col_left, col_mid, col_right = st.columns([2, 1, 1])
    with col_left:
        # session_stateに現在のファイルインデックスを保持
        if "tip_index" not in st.session_state:
            st.session_state["tip_index"] = 0
        if st.session_state["tip_index"] >= len(md_files):
            st.session_state["tip_index"] = 0

        file_labels = [f.replace(".md", "") for f in md_files]
        selected_label = st.selectbox(
            "ドキュメントを選択",
            file_labels,
            index=st.session_state["tip_index"],
            key="tip_selector",
            help="表示するTipsを選択してください"
        )
        st.session_state["tip_index"] = file_labels.index(selected_label)

    st.markdown("---")

    # 選択されたMDファイルを読み込んで表示
    current_file = md_files[st.session_state["tip_index"]]
    file_path = os.path.join(DOCS_DIR, current_file)

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # ファイル情報
        file_stat = os.stat(file_path)
        char_count = len(content)
        from datetime import datetime
        modified_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M")

        st.caption(f"📄 {current_file}  |  文字数: {char_count:,}  |  更新: {modified_time}")

        # 区切り線付きで本文を表示
        with st.container():
            st.markdown(content)

        # ダウンロードボタン
        st.download_button(
            label="📥 このドキュメントをダウンロード",
            data=content.encode("utf-8"),
            file_name=current_file,
            mime="text/markdown",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"ドキュメントの読み込みに失敗しました: {e}")


# ------------------------------------------------
# メインエントリポイント
# ------------------------------------------------
def main():
    st.set_page_config(page_title="LocalNote", page_icon=":shark:")

    # 状態の初期化
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = load_config(mode="model")
    if "config" not in st.session_state:
        st.session_state["config"] = load_config(mode="config")
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = load_config(mode="temperature")
    if "max_tokens" not in st.session_state:
        st.session_state["max_tokens"] = load_config(mode="max_tokens")
    if "rag_engine" not in st.session_state:
        st.session_state["rag_engine"] = RAGEngine(
            st.session_state["config"],
            temperature=st.session_state["temperature"],
            max_tokens=st.session_state["max_tokens"]
        )
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

    # ------------------------------------------------------------
    # サイドバー
    # ------------------------------------------------------------
    with st.sidebar:
        st.title("📒LocalNote")

        # メニュー（Tipsを追加）
        menu_options = ["💭チャット画面", "📖 Tips/ヘルプ", "⚙️設定画面"]
        app_mode = st.radio("メニュー", menu_options)

        if st.session_state["db_ready"] and app_mode != "📖 Tips/ヘルプ":
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
                    params = load_rag_params(db)
                    if params:
                        st.session_state["rag_engine"].add_to_database(
                            temp_file_dir, db,
                            params["chunk_size"], params["chunk_overlap"]
                        )

                        final_file_dir = os.path.join(PDF_DIR, os.path.basename(st.session_state["current_db"]))
                        for file_name in os.listdir(temp_file_dir):
                            shutil.move(os.path.join(temp_file_dir, file_name), os.path.join(final_file_dir, file_name))
                        shutil.rmtree(temp_file_dir, ignore_errors=True)
                        st.success("✅ファイルの追加が完了しました")

            st.subheader("🗂 現在の学習ファイル")
            db_name = os.path.basename(st.session_state["current_db"])
            file_dir = os.path.join(PDF_DIR, db_name)

            if os.path.exists(file_dir):
                existing_files = sorted(os.listdir(file_dir))
                selected_files = []
                is_delete_chat = False

                if existing_files:
                    for f in existing_files:
                        checked = st.checkbox(f, key=f"del_{db_name}_{f}")
                        if checked:
                            selected_files.append(f)

                    if selected_files:
                        if len(existing_files) == len(selected_files):
                            is_delete_chat = True
                            st.warning("すべてのファイルが選択されているため，チャットが削除されます")
                        else:
                            st.warning("選択したファイルはデータベースから完全に削除されます")

                        if st.button("❌ 選択したファイルを削除"):
                            with st.spinner("削除中..."):
                                if is_delete_chat:
                                    delete_chat(st.session_state["current_db"], os.path.join(OUTPUT_DIR, db_name))
                                else:
                                    full_paths = [os.path.join(file_dir, f) for f in selected_files]
                                    st.session_state["rag_engine"].delete_files_from_database(
                                        st.session_state["current_db"], full_paths
                                    )
                                    for path in full_paths:
                                        if os.path.exists(path):
                                            os.remove(path)
                                    st.success("✅ 選択した学習ファイルを削除しました")
                                    time.sleep(0.2)
                                    st.rerun()
                else:
                    st.info("学習ファイルがありません")
            else:
                st.info("学習フォルダが存在しません")

        # 単語帳
        st.markdown("---")
        st.subheader("📝 単語帳データの出力")
        num_words = st.slider("単語の数", min_value=5, max_value=40, value=20)
        if st.session_state["db_ready"]:
            db_name = os.path.basename(st.session_state["current_db"])
            if st.button("単語帳データを生成"):
                with st.spinner("AIが用語を抽出中...\n(数分かかることがあります)"):
                    try:
                        st.session_state["flashcard_csv"] = st.session_state["rag_engine"].generate_flashcard(
                            st.session_state["current_db"], num_words
                        )
                        st.success("✅ ファイルを作成しました！")
                    except Exception as e:
                        st.error(f"生成中にエラーが発生しました: {e}")

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

        # チャット管理
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
                    params = load_rag_params(target_db_path)
                    if params:
                        st.session_state["rag_engine"].k = params["k"]
                    st.session_state["history"] = st.session_state["history_manager"].load(selected_db)
                    st.rerun()
            else:
                st.warning("既存のチャットがありません。'新規チャットの作成'から作成してください。")
        else:
            new_db_name = st.text_input("新しいチャット名を入力")
            can_exe = True
            if new_db_name:
                target_db_path = os.path.join(DB_DIR, new_db_name)
                if os.path.isdir(target_db_path):
                    st.warning("このチャット名は使用できません。他のチャット名にしてください。")
                    can_exe = True
                else:
                    can_exe = False

            files = st.file_uploader(
                "まとめたいファイルのアップロード",
                accept_multiple_files=True,
                type=["pdf", "md", "docx", "xlsx", "pptx"],
                disabled=can_exe
            )

            if st.button("作成して実行する"):
                if files and new_db_name:
                    with st.spinner("実行中..."):
                        save_dir = os.path.join(PDF_DIR, new_db_name)
                        os.makedirs(save_dir, exist_ok=True)
                        for file in files:
                            file_path = os.path.join(save_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())

                        st.session_state["history"] = {}
                        total_chars = st.session_state["rag_engine"].estimate_total_chars(save_dir)
                        chunk_size, chunk_overlap, k_value = auto_rag_params(total_chars)

                        st.session_state["rag_engine"].build_database(save_dir, target_db_path, chunk_size, chunk_overlap, k_value)
                        save_rag_params(target_db_path, chunk_size, chunk_overlap, k_value)

                        st.session_state["db_ready"] = True
                        st.session_state["current_db"] = target_db_path
                        st.success("✅データベースの構築が完了しました！")

    # ------------------------------------------------------------
    # メイン画面
    # ------------------------------------------------------------
    if app_mode == "💭チャット画面":
        render_chat_page()
    elif app_mode == "📖 Tips/ヘルプ":
        render_tips_page()
    elif app_mode == "⚙️設定画面":
        render_settings_page()


# ------------------------------------------------
# チャット画面
# ------------------------------------------------
def render_chat_page():
    if st.session_state["db_ready"]:
        db_name = os.path.basename(st.session_state["current_db"])
        st.success(f"チャット接続中: [**{db_name}**], モデル: [**{st.session_state['model_name']}**]")

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
                            st.session_state["current_db"], user_query, st.session_state["history"]
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        chunk_size_display = 5
                        for i in range(0, len(answer), chunk_size_display):
                            dislpay_char += answer[i:i + chunk_size_display]
                            placeholder.markdown(dislpay_char + "▌")
                            time.sleep(.005)

                        if sources:
                            st.markdown("---")
                            st.markdown("📚 **参考にした資料:**")
                            for src in sources:
                                st.markdown(f"- {src}")

                        data = {
                            "input": {"role": "user", "content": user_query},
                            "output": {"role": "assistant", "content": answer}
                        }
                        st.session_state["history"]["question_" + str(len(st.session_state["history"]))] = data
                        st.session_state["history_manager"].save(db_name, st.session_state["history"])

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
    else:
        st.info("チャットを開始するにはサイドバーからチャットを選択してください。")


# ------------------------------------------------
# 設定画面
# ------------------------------------------------
def render_settings_page():
    st.title("設定")

    # --- モデル + 生成パラメータ設定 ---
    with st.expander("🤖 モデル設定", expanded=True):
        st.markdown("使用するモデルと生成パラメータを設定できます。")
        col_left, col_right = st.columns(2)
        with col_left:
            model_name = st.selectbox(
                "モデル",
                ["gemma4:e4b", "gemma4:e2b", "その他"],
                help="使用するOllamaモデルを選択"
            )
            if model_name == "その他":
                model_name = st.text_input("モデル名を入力")
        with col_right:
            temperature = st.slider(
                "Temperature（創造性）",
                min_value=0.0, max_value=2.0,
                value=float(st.session_state["temperature"]),
                step=0.1,
                help="低いほど正確・安定した回答、高いほど多様で創造的な回答に\nデフォルト: 0"
            )
            max_tokens = st.slider(
                "最大トークン数（回答長）",
                min_value=512, max_value=8192,
                value=st.session_state["max_tokens"],
                step=256,
                help="1回の回答で生成する最大トークン数\nデフォルト: 4096"
            )

    # --- コンテキスト内学習（プロンプト）設定 ---
    with st.expander("📝 コンテキスト内学習設定", expanded=True):
        st.markdown("AIへの指示（システムプロンプト）を編集できます。"
                    "`{context}` に検索された参考情報が挿入されます。")

        if "applied_preset" not in st.session_state:
            st.session_state["applied_preset"] = None

        preset_col1, preset_col2 = st.columns([3, 1])
        with preset_col1:
            preset_names = list(PROMPT_PRESETS.keys())
            selected_preset = st.selectbox(
                "プロンプトプリセット",
                ["（カスタム）"] + preset_names,
                key="preset_selector",
                help="プリセットを選択するとエディタの内容が置き換わります（自動保存はされません）"
            )
        with preset_col2:
            st.markdown("###")
            if st.button("🔄 デフォルトに戻す", use_container_width=True):
                st.session_state["config"] = DEFAULT_PROMPT
                st.session_state["applied_preset"] = None
                st.session_state["preset_selector"] = "（カスタム）"
                st.rerun()

        if selected_preset != "（カスタム）":
            if st.session_state.get("applied_preset") != selected_preset:
                st.session_state["config"] = PROMPT_PRESETS[selected_preset]
                st.session_state["applied_preset"] = selected_preset
                st.rerun()
        else:
            if st.session_state.get("applied_preset") is not None:
                st.session_state["applied_preset"] = None
                st.rerun()

        new_prompt = st.text_area(
            "システムプロンプト",
            value=st.session_state["config"],
            height=350,
            help="AIへの指示を自由に記述してください。\n【参考情報】セクションが自動的に追加されます。"
        )

        st.caption(f"現在の文字数: {len(new_prompt)} 文字")

        with st.expander("👁️ 実際にAIに送信される全文をプレビュー"):
            full_prompt = new_prompt + "\n\n---\n【参考情報】\n" + (
                "【資料1: example.pdf】\n...（ここに検索されたチャンクが挿入されます）...\n\n"
                "---\n【資料2: notes.md】\n..."
            )
            st.code(full_prompt, language="markdown", line_numbers=True)

        if st.button("💾 設定を保存", type="primary", use_container_width=True):
            st.session_state["config"] = new_prompt
            st.session_state["model_name"] = model_name
            st.session_state["temperature"] = temperature
            st.session_state["max_tokens"] = max_tokens
            st.session_state["applied_preset"] = None
            save_config(new_prompt, model_name, temperature, max_tokens)
            st.session_state["rag_engine"].update_prompt(new_prompt)
            st.session_state["rag_engine"].update_llm(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
            st.success("✅ 設定を保存しました")

    st.markdown("---")

    # --- 検索設定 ---
    with st.expander("🔍 検索精度設定", expanded=False):
        st.markdown("MMR検索・Reranking・Fair Searchで検索結果の精度・多様性を向上させます。")
        fair_search = st.checkbox(
            "⚖️ Fair Search（ソースファイル間の公平検索）",
            value=st.session_state["rag_engine"].fair_search,
            help="有効にすると、各ソースファイルから均等にチャンクを取得します。\n"
                 "大きいファイルの情報に偏らず、小さいファイルの情報も考慮されるようになります。"
        )
        col1, col2 = st.columns(2)
        with col1:
            retriever_type = st.selectbox(
                "検索戦略",
                ["similarity", "mmr", "hybrid"],
                index=["similarity", "mmr", "hybrid"].index(st.session_state["rag_engine"].retriever_type),
                help="similarity: 標準的な関連性検索\nmmr: 多様性と関連性のバランス\nhybrid: 複合検索"
            )
        with col2:
            enable_reranking = st.checkbox(
                "📊 Reranking有効化",
                value=st.session_state["rag_engine"].enable_reranking,
                help="検索結果をより正確にスコアリングし直します"
            )

        if enable_reranking:
            rerank_model = st.selectbox(
                "Rerankerモデル",
                ["bge", "cross-encoder"],
                index=["bge", "cross-encoder"].index(
                    st.session_state["rag_engine"].rerank_model
                    if st.session_state["rag_engine"].rerank_model in ["bge", "cross-encoder"]
                    else "bge"
                ),
                help="bge: 軽量で高速\ncross-encoder: より高精度（要GPU推奨）"
            )
        else:
            rerank_model = "noop"

        if retriever_type == "mmr":
            mmr_lambda = st.slider(
                "🎯 MMR多様性・関連性バランス",
                min_value=0.0, max_value=1.0,
                value=st.session_state["rag_engine"].mmr_lambda_mult,
                step=0.1,
                help="0.0 → 多様性重視\n1.0 → 関連性重視"
            )
        else:
            mmr_lambda = st.session_state["rag_engine"].mmr_lambda_mult

        # 会話履歴数の制限
        max_history_pairs = st.slider(
            "💬 History-aware参照履歴数",
            min_value=1,
            max_value=20,
            value=st.session_state["rag_engine"].max_history_pairs,
            step=1,
            help="質問再構築時に考慮する過去の会話ペア数。\n"
                  "多いほど文脈を理解できるが、トークン消費が増加。\n"
                  "推奨: 3〜10"
        )

        # Smart Forgetting: 重要会話を保持、不要会話を忘却
        smart_forgetting = st.checkbox(
            "🧠 Smart Forgetting（重要度ベース忘却）",
            value=st.session_state["rag_engine"].smart_forgetting,
            help="有効にすると、重要な会話（専門的な質問・長文の議論）は長く保持し、\n"
                  "重要でない会話（挨拶・短い確認・感謝など）は優先的に忘却します。\n"
                  "同じ履歴数でも、記憶すべき情報がより多く残ります。"
        )

        st.session_state["rag_engine"].retriever_type = retriever_type
        st.session_state["rag_engine"].enable_reranking = enable_reranking
        st.session_state["rag_engine"].rerank_model = rerank_model
        st.session_state["rag_engine"].mmr_lambda_mult = mmr_lambda
        st.session_state["rag_engine"].fair_search = fair_search
        st.session_state["rag_engine"].max_history_pairs = max_history_pairs
        st.session_state["rag_engine"].smart_forgetting = smart_forgetting

        fair_badge = " + ⚖️Fair Search" if fair_search else ""
        smart_badge = " + 🧠SmartForgetting" if smart_forgetting else ""
        st.info(f"✅ 現在の設定: {retriever_type.upper()} 検索"
                + (f" + {rerank_model} Reranking" if enable_reranking else "") +
                f" | 履歴: {max_history_pairs}ペア"
                + smart_badge
                + fair_badge)

    st.markdown("---")

    # --- データ管理 ---
    with st.expander("🗄️ データ管理", expanded=False):
        tab1, tab2 = st.tabs(["🗑️ 履歴削除", "🚨 チャット削除"])
        with tab1:
            st.markdown("チャットの会話履歴のみを削除します（データベースは維持）。")
            if st.session_state["db_ready"]:
                db_name = os.path.basename(st.session_state["current_db"])
                st.warning(f"対象: **{db_name}**")
                if st.button("履歴を削除", use_container_width=True):
                    st.session_state["history_manager"].clear(db_name)
                    st.session_state["history"] = {}
                    st.success("✅ 履歴を削除しました")
                    time.sleep(0.3)
                    st.rerun()
            else:
                st.info("サイドバーからチャットを読み込んでください。")
        with tab2:
            st.markdown("**チャット全体（データベース・履歴・アップロードファイル）を完全に削除します。**")
            st.markdown("この操作は元に戻せません。")
            if st.session_state["db_ready"]:
                db_name = os.path.basename(st.session_state["current_db"])
                st.error(f"⚠️ 削除対象: **{db_name}**")
                if st.button("🗑️ すべて削除", type="primary", use_container_width=True):
                    delete_chat(st.session_state["current_db"], os.path.join(OUTPUT_DIR, db_name))
            else:
                st.info("サイドバーからチャットを読み込んでください。")


if __name__ == "__main__":
    main()
