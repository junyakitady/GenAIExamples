import os
from operator import itemgetter
import streamlit as st
import streamlit.components.v1 as components
import langchain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import VertexAI
import vertexai

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "asia-northeast1"
DATASTORE_ID = os.environ.get("DATASTORE_ID")
vertexai.init(project=PROJECT_ID, location=REGION)

st.title("Chat with Vertex AI Search")
st.text("Vertex AI Search に登録済みの文書でチャットできます。ChatPDFのサンプルと同じPDFが登録されています。")
with st.sidebar:
    components.iframe("https://storage.googleapis.com/public4llm/es/index.html", height=650)

if "esmessages" not in st.session_state:
    st.session_state.esmessages = []

    # Text model instance integrated with LangChain
    llm = VertexAI(model_name="gemini-1.5-flash", max_output_tokens=2048, temperature=0.5, verbose=True)
    # Vertex AI Search retriever
    retriever = VertexAISearchRetriever(project_id=PROJECT_ID, data_store_id=DATASTORE_ID,
                                        get_extractive_answers=True, max_extractive_answer_count=3, max_documents=3)
    # Create chain to answer questions
    template = """次のコンテキスト情報を利用して、最後の質問に答えてください。回答は300字程度で回答してください。:
Context: {context}

Question: {question}"""
    prompt = PromptTemplate.from_template(template)
    st.session_state.esqa = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# Display chat messages from history on app rerun
for esmessages in st.session_state.esmessages:
    with st.chat_message(esmessages["role"]):
        st.markdown(esmessages["content"])

# Accept user input
if query := st.chat_input("日本語変換の確定でサブミットされるため、質問はペーストしてください。"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # RetrievalQA
    esqa = st.session_state.esqa
    response = esqa.invoke(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add message to chat history
    st.session_state.esmessages.append({"role": "user", "content": query})
    st.session_state.esmessages.append({"role": "assistant", "content": response})
