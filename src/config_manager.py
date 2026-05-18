"""
Configuration management for LocalNote.
Handles loading/saving of system prompts, model settings, and RAG parameters.
"""

import os, json

CONFIG_DIR = "./config.json"

DEFAULT_PROMPT = """あなたは学習教材の内容をわかりやすく解説するアシスタント「LocalNote」です。

## 基本ルール
- 以下の【参考情報】のみを根拠に回答してください。
- 参考情報にない内容は「資料に記載がありません」と正直に伝えてください。
- 回答は簡潔かつ構造的に、日本語で記述してください。

## 回答の品質
- 情報をそのまま抜き出すだけでなく、必要に応じて箇条書き・見出し・テーブル等を使って整理してください。
- 複数の資料にまたがる情報は統合して説明してください。
- 数値データや固有名詞は正確に引用してください。
- 資料の該当箇所が不明瞭な場合も、「〜と解釈できます」など曖昧さを明示してください。

## 禁止事項
- 指示がない限り、テスト要項や成績についてはまとめないでください。
- 参考情報にない知識を補完しないでください。
- 不確かな情報をあたかも確実であるかのように述べないでください。
"""

DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 4096

PROMPT_PRESETS = {
    "デフォルト（構造志向）": DEFAULT_PROMPT,
    "簡潔志向": """あなたは学習教材の内容を簡潔に答えるアシスタントです。

## ルール
- 以下の【参考情報】のみを根拠に回答してください。
- 回答は3文以内で簡潔に。ただし必要な情報は全て含めてください。
- 参考情報にない場合は「資料に記載がありません」と伝えてください。
- テスト要項や成績については、指示がない限りまとめないでください。
""",
    "詳細解説志向": """あなたは学習教材の内容を丁寧に解説するアシスタント「LocalNote」です。

## 基本ルール
- 以下の【参考情報】のみを根拠に回答してください。
- 参考情報にない内容は「資料に記載がありません」と正直に伝えてください。

## 解説スタイル
- 最初に一言で結論を述べ、その後詳細な解説を加えてください。
- 必要に応じて具体例や図表の説明を補って理解を助けてください。
- 複数の視点から情報を整理し、学習者が体系的に理解できるようにしてください。
- 専門用語が登場したら、その都度簡単な説明を添えてください。

## 禁止事項
- 指示がない限り、テスト要項や成績についてはまとめないでください。
- 参考情報にない知識を補完しないでください。
""",
    "要約志向": """あなたは資料を要約するアシスタントです。

## ルール
- 以下の【参考情報】のみを根拠に要約してください。
- 箇条書きを使って重要なポイントを過不足なく列挙してください。
- 各ポイントは1〜2行で簡潔に。
- 必要に応じて小見出しでグループ化してください。
- 参考情報にない内容は含めないでください。
- 指示がない限り、テスト要項や成績は要約に含めないでください。
""",
}


def load_config(mode="config"):
    """Load configuration from JSON file."""
    defaults = {
        "config": DEFAULT_PROMPT,
        "model": "gemma4:e4b",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS
    }
    if os.path.exists(CONFIG_DIR):
        try:
            with open(CONFIG_DIR, "r", encoding="utf-8") as f:
                data = json.load(f)
            if mode in data:
                return data[mode]
            return defaults.get(mode)
        except (json.JSONDecodeError, KeyError):
            return defaults.get(mode)
    return defaults.get(mode)


def save_config(config_data, model_name, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
    """Save configuration to JSON file."""
    data = {
        "config": config_data,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if os.path.exists(CONFIG_DIR):
        try:
            with open(CONFIG_DIR, "r", encoding="utf-8") as f:
                existing = json.load(f)
                existing.update(data)
                data = existing
        except (json.JSONDecodeError, Exception):
            pass
    with open(CONFIG_DIR, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_rag_params(db_path, chunk_size, chunk_overlap, k):
    """Save RAG chunk parameters to JSON file inside db_path."""
    os.makedirs(db_path, exist_ok=True)
    params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k
    }
    with open(os.path.join(db_path, "rag_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)


def load_rag_params(db_path):
    """Load RAG chunk parameters from JSON file inside db_path."""
    param_path = os.path.join(db_path, "rag_params.json")
    if os.path.exists(param_path):
        with open(param_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
