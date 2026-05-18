# LangChain — LLMアプリケーション構築フレームワーク

**LangChain** はLLMを使ったアプリケーションを構築するためのフレームワークです。
LocalNoteではRAGパイプラインの構築に使用しています。

## 使用している主なコンポーネント

### ChatPromptTemplate
- システムプロンプト + チャット履歴 + ユーザー入力をテンプレート化
- コンテキスト内学習（プロンプトのカスタマイズ）を実現

### create_history_aware_retriever
- チャット履歴を考慮して質問を言い換え
- 「それってどういう意味？」→「○○理論の意味は？」

### create_stuff_documents_chain
- 検索結果のドキュメントをプロンプトに「詰め込む」(stuff)
- ドキュメントごとに `【ファイル名】` タグを付けて出典を明示

### create_retrieval_chain
- 検索 → 生成を1つのチェーンに連結
- `rag_chain.invoke()` で一発実行

### HuggingFaceEmbeddings
- `intfloat/multilingual-e5-base` モデルを使用
- 日本語に対応した高品質な埋め込みベクトルを生成

## LangChain Classic について

LocalNoteは `langchain_classic` パッケージを使用しています。
これは従来のLangChain（v0.1.x系）のチェーンAPIを継続して使うための互換パッケージです。
