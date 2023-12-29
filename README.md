# PDF 文書の内容にもとづいて回答するボットを作成する
## はじめに
このリポジトリは Vertex AI の生成 AI を利用し、Cloud Run 上で動作する Python アプリケーションのサンプル プログラムを提供します。
ページ別に以下の機能を提供しています。
1. PDF をアップロードし、その内容に回答できるボット
  - Notebook は[こちら](./notebook/DocQA_PaLM_LangChain.ipynb)
2. Vertex AI Search にアップロード済みの PDFに対し、その内容に回答できるボット
  - Notebook は[こちら](./notebook/DocQA_VertexAISearch.ipynb)
3. Gemini、PaLM Bison、PaLM Unicorn に同時のプロンプトを投げて比較

以下のプロダクトを利用しています。
- [Vertex AI PaLM API](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview)
- [LangChain 🦜️🔗](https://python.langchain.com/docs/get_started/introduction.html)
- [Streamlit](https://streamlit.io/)

コード中の`PROJECT_ID`や`DATASTORE_ID`などの変数は各自環境の値で置き換えてください。

## ローカルでの実行
ローカルで実行する場合は、Google Cloud の `Vertex AI ユーザー`権限を保持するサービスアカウントキーを作成し、キーファイルを保存しておきます。

```shell
export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
```
Python v3.8.1 以降が導入されていることを確認し、以下を実行します。 
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ChatPDF.py
```
## Cloud Run での実行

Cloud Console の Cloud Run のサービス作成 UI を利用する場合、
- ソースリポジトリからビルドタイプ Docker ファイルでビルド
- 認証に未認証の呼び出しを許可
- セキュリティのサービスアカウントに、`Vertex AI ユーザー`権限を保持するサービスアカウントを指定

を各自の環境用に指定すれば自動でデプロイ、動作します。
