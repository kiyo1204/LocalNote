"""
LocalNote - エントリーポイント
Streamlitアプリケーションの起動は src/app.py に委譲します。
"""
import os

# プロジェクトルートを作業ディレクトリに設定
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.app import main

if __name__ == "__main__":
    main()
