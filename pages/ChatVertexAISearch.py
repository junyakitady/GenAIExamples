import time
import streamlit as st
import langchain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import VertexAI
from langchain.retrievers import GoogleVertexAISearchRetriever
import vertexai

# init Vertex AI
PROJECT_ID = "<your_project_id>"
REGION = "us-central1"
DATASTORE_ID = "<datastore_id>"
vertexai.init(project=PROJECT_ID, location=REGION)

st.title("Chat with Vertex AI Search")
st.text("Vertex AI Search に登録済みの文書でチャットできます。ChatPDFのサンプルと同じPDFが登録されています。")
with st.sidebar:
    st.components.v1.iframe("https://storage.googleapis.com/public4llm/es/index.html", height=650)

if "esmessages" not in st.session_state:
    st.session_state.esmessages = []

    # Text model instance integrated with LangChain
    llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8, verbose=True)
    # Vertex AI Search retriever
    retriever = GoogleVertexAISearchRetriever(project_id=PROJECT_ID, search_engine_id=DATASTORE_ID,
                                              get_extractive_answers=True, max_extractive_answer_count=3, max_documents=3)
    # Create chain to answer questions
    st.session_state.esqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # st.session_state.esqa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,)

# Display chat messages from history on app rerun
for esmessages in st.session_state.esmessages:
    with st.chat_message(esmessages["role"]):
        st.markdown(esmessages["content"])

# Accept user input
if prompt := st.chat_input("日本語変換の確定でサブミットされるため、質問はペーストしてください。"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # RetrievalQA
    esqa = st.session_state.esqa
    response = esqa({"query": prompt})  # "query" for RetrievalQA "question" for ConversationalRetrievalChain

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response["result"].split():  # "result" for RetrievalQA "answer" for ConversationalRetrievalChain
            full_response += chunk + " "
            time.sleep(0.1)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add message to chat history
    st.session_state.esmessages.append({"role": "user", "content": prompt})
    st.session_state.esmessages.append({"role": "assistant", "content": response["result"]})
