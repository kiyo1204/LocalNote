"""
Utility functions for LocalNote.
Includes JSON extraction, parameter auto-tuning, and chat deletion.
"""

import os, re, shutil, gc, time
import chromadb
import json5


def _extract_json(text):
    """
    Extract and parse JSON from potentially messy LLM output.
    Uses json5 for lenient parsing (handles trailing commas, unquoted keys,
    single quotes, comments, raw newlines in strings, etc.), then falls back
    to regex-based extraction with cleaning.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # ---- 試行1: json5 で直接パース ----
    # json5 は以下を許容:
    #   - 文字列内の raw 改行
    #   - 末尾カンマ
    #   - シングルクォート文字列
    #   - キーのクォート省略
    #   - コメント (//, /* */)
    try:
        return json5.loads(text)
    except (ValueError, TypeError):
        pass

    # ---- 試行2: 前処理（raw改行除去）→ json5 ----
    cleaned = _clean_json_string_newlines(text)
    try:
        return json5.loads(cleaned)
    except (ValueError, TypeError):
        pass

    # ---- 試行3: JSONオブジェクト {...} を抽出してパース ----
    parsed = _extract_json_block(text, r'\{.*\}')
    if parsed is not None:
        return parsed

    # ---- 試行4: JSON配列 [...] を抽出してパース ----
    parsed = _extract_json_block(text, r'\[.*\]')
    if parsed is not None:
        return parsed

    return None


def _extract_json_block(text, pattern):
    """Extract a JSON block matching the given regex pattern and attempt to parse it."""
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    block = match.group()
    # json5 で直接パース
    try:
        return json5.loads(block)
    except (ValueError, TypeError):
        pass
    # 改行クリーニング後に再試行
    cleaned = _clean_json_string_newlines(block)
    try:
        return json5.loads(cleaned)
    except (ValueError, TypeError):
        pass
    return None


def _clean_json_string_newlines(text):
    """
    JSON文字列値内の raw 改行（\n, \r）をスペースに置換する。
    LaTeX数式を含むLLM出力などで頻発する問題に対処。
    エスケープシーケンス（\" など）を正しく考慮したステートマシン。
    """
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and (ch == '\n' or ch == '\r'):
            result.append(' ')
            continue
        result.append(ch)
    return ''.join(result)


def auto_rag_params(total_chars: int):
    """
    Automatically determine chunk size and K value based on total character count.
    """
    if total_chars < 5_000:
        chunk_size = 300
        k = 20
    elif total_chars < 20_000:
        chunk_size = 500
        k = 18
    elif total_chars < 50_000:
        chunk_size = 800
        k = 15
    elif total_chars < 100_000:
        chunk_size = 1000
        k = 10
    elif total_chars < 300_000:
        chunk_size = 1200
        k = 8
    else:
        chunk_size = 1500
        k = 6

    chunk_overlap = int(chunk_size * 0.2)
    return chunk_size, chunk_overlap, k


def delete_chat_files(db_name, history_name, pdf_dir):
    """
    Delete chat data (database, history, uploaded files) with retry logic.
    Intended to be called from within a Streamlit context (uses st.rerun).
    """
    import streamlit as st

    # 状態の初期化
    st.session_state["db_ready"] = False
    st.session_state["current_db"] = ""
    st.session_state["history"] = {}

    # 1. RAGEngineのキャッシュからChromaインスタンスを解放
    if "rag_engine" in st.session_state:
        st.session_state["rag_engine"].dispose_db(db_name)

    # 2. ChromaDBのシステムキャッシュをクリア
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    # 3. ガベージコレクション
    gc.collect()
    time.sleep(0.3)

    try:
        # 4. 履歴・データベースの削除（リトライ付き）
        for path in [db_name, history_name]:
            if os.path.exists(path):
                for attempt in range(3):
                    try:
                        shutil.rmtree(path, ignore_errors=False)
                        break
                    except (PermissionError, OSError):
                        if attempt < 2:
                            time.sleep(0.5)
                        else:
                            shutil.rmtree(path, ignore_errors=True)

        # PDFの削除
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir, ignore_errors=True)

        st.success("削除しました。")
        time.sleep(0.2)
        st.rerun()
    except Exception as e:
        st.error(f"削除中にエラーが発生しました: {e}")
