import os
import streamlit as st
import langchain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import VertexAI
import vertexai
import bigframes
import bigframes.pandas as bpd

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "asia-northeast1"
BQ_LOCATION = os.environ.get("BQ_LOCATION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_TABLE = os.environ.get("BQ_TABLE")
vertexai.init(project=PROJECT_ID, location=REGION)

# layoout page
st.title("Ask BigQuery")
st.text("2023年度 公示地価情報を保持するテーブルに自然言語で質問できます。")

with st.form('my_form'):
    prompt = st.text_area(label='Prompt', value="""住所が東京都杉並区の中で地価を高い順に5個教えてください。
SQLの前後にMarkdown記号は含まないこと。
以下のフォーマットでMarkdown表形式で日本語で回答してください。
最寄り駅 | 住所 | 地価 | 値上がり率""")
    submitted = st.form_submit_button('Submit')

# Text model instance integrated with LangChain
llm = VertexAI(model_name="gemini-1.5-pro", max_output_tokens=2048, temperature=0.5)

db = SQLDatabase.from_uri("bigquery://"+PROJECT_ID+"/"+BQ_DATASET)
sqlagent = create_sql_agent(llm=llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm), verbose=True, max_iterations=10)

if submitted:
    response = sqlagent.invoke(prompt)
    st.text("Output")
    st.markdown(response["output"])
