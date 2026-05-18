# Streamlit — UIフレームワーク

**Streamlit** はPythonだけでWebアプリケーションが作れるフレームワークです。
LocalNoteのチャット画面・設定画面・Tips表示はすべてStreamlitで構築されています。

## 特徴

- **Pythonオンリー**: HTML/CSS/JavaScriptを書かずにPythonコードだけでUIを作成
- **リアクティブ**: 変数を変更するだけで自動的にUIが再描画される
- **ウィジェット充実**: チャット入力・スライダー・セレクトボックス・ファイルアップローダーなど

## LocalNoteで使っている機能

| 機能 | 使用箇所 |
|------|----------|
| `st.chat_input` / `st.chat_message` | チャット画面 |
| `st.sidebar` | サイドバーメニュー・ファイル管理 |
| `st.expander` / `st.tabs` | 設定画面のセクション整理 |
| `st.file_uploader` | ファイルアップロード |
| `st.progress` / `st.spinner` | 処理中表示 |
| `st.dialog` | 削除確認ダイアログ |
| `st.session_state` | 状態管理（モデル設定・履歴など） |
| `st.rerun` | 画面の再描画トリガー |

## なぜStreamlitか？

- **プロトタイプ〜本番まで**: 1日あればRAGアプリが作れる
- **デプロイ容易**: `streamlit run main.py` だけで起動
- **コミュニティ豊富**: 拡張機能やテンプレートが多数
