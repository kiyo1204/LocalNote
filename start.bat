@echo off
chcp 65001 > nul
cd /d %~dp0

echo === LocalNote 起動処理開始 ===

REM 仮想環境がなければ作成
if not exist ".venv\Scripts\activate.bat" (
    echo 仮想環境を作成します...
    python -m venv .venv
)

REM 仮想環境を有効化
call .venv\Scripts\activate

REM pip 更新
python -m pip install --upgrade pip

REM requirements.txt があれば install
if exist "requirements.txt" (
    echo 依存関係をインストール中...
    pip install -r requirements.txt
) else (
    echo requirements.txt が見つかりません
)

REM アプリ起動
echo アプリを起動します...
streamlit run main.py

pause