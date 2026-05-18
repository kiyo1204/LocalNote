# Ollama — ローカルLLM実行環境

**Ollama** はローカルPCで大規模言語モデル（LLM）を手軽に実行できるツールです。
LocalNoteはOllamaを通じてLLMと通信しています。

## 特徴

- **完全ローカル**: インターネット不要、データが外部に出ない
- **プライバシー保護**: 学習データが外部サーバーに送信されない
- **豊富なモデル**: Gemma・Llama・Mistral・Qwen など多数対応
- **軽量運用**: APIサーバーとして常駐し、HTTPで通信

## LocalNoteとの連携

```
LocalNote (main.py)
    │  HTTPリクエスト（LangChain経由）
    ▼
Ollama API (localhost:11434)
    │  モデル推論
    ▼
LLMモデル（例: gemma4）
```

## 推奨モデル

| モデル | サイズ | 特徴 |
|--------|--------|------|
| **gemma4:e4b** | ~4GB | 高精度、LocalNote推奨 |
| **gemma4:e2b** | ~2GB | 軽量、低スペックPC向け |
| llama3.2 | ~3GB | バランス良好 |
| qwen2.5 | ~4GB | 日本語性能優秀 |

## インストール

```bash
# 1. Ollamaをインストール（https://ollama.com）
# 2. モデルをダウンロード
ollama pull gemma4:e4b

# 3. サーバー起動（自動で常駐）
# 4. LocalNoteを起動
```

## Tips

- **Temperature**: 0（正確）〜2.0（創造的）。事実系の質問は0推奨
- **1回の推論**: モデルサイズやPCスペックにより10秒〜1分程度
