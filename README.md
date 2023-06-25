# PDF 文書の内容にもとづいて回答するボットを作成する
## はじめに
このリポジトリは以下のサンプル プログラムを提供します。
- Cloud Run 上で動作する Python アプリケーション
- 同じアプリケーションの流れを Notebook で解説したもの
  - Notebook は[こちら](./notebook/DocQA_PaLM_LangChain.ipynb)

以下のプロダクトを利用しています。
- [Vertex AI PaLM API](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview)
- [LangChain 🦜️🔗](https://python.langchain.com/docs/get_started/introduction.html)
- [Streamlit](https://streamlit.io/)

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


## 参考情報
- [Getting Started with LangChain 🦜️🔗 + Vertex AI PaLM API](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/langchain-intro/intro_langchain_palm_api.ipynb)
- [Question Answering with Large Documents using LangChain 🦜🔗](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/document-qa/question_answering_large_documents_langchain.ipynb)
