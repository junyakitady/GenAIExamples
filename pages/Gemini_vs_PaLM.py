import os
import streamlit as st
import langchain
from langchain_google_vertexai import VertexAI
import vertexai
import asyncio

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# layoout page
st.set_page_config(layout="wide")
st.title("Vertex AI の言語モデルを比較")
with st.form('my_form'):
    prompt = st.text_area(label='Prompt', value='小説好きの人にお勧めのロンドンの名所を教えてください')
    submitted = st.form_submit_button('Submit')

col1, col2, col3 = st.columns(3)
col1.subheader('Gemini Pro')
col2.subheader('PaLM Bison')
col3.subheader('PaLM Unicorn')
output_tokens = st.sidebar.text_input('max_output_tokens', value=1024)
temperature = st.sidebar.text_input('temperature', value=0.2)

# Text model instance integrated with LangChain
bison = VertexAI(model_name="text-bison", max_output_tokens=output_tokens, temperature=temperature)
unicorn = VertexAI(model_name="text-unicorn", max_output_tokens=output_tokens, temperature=temperature)
geminip = VertexAI(model_name="gemini-pro", max_output_tokens=output_tokens, temperature=temperature)


async def get_answer(llm, col):
    ans = await llm.ainvoke(prompt)
    col.markdown(ans)


async def ask_llms():
    tasks = []
    llms = [geminip, bison, unicorn]
    cols = [col1, col2, col3]
    for index in range(len(llms)):
        tasks.append(asyncio.create_task(get_answer(llms[index], cols[index])))

    await asyncio.gather(*tasks)

if submitted:
    asyncio.run(ask_llms())
