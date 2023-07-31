import time
import streamlit as st
import langchain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import VertexAI
from langchain.retrievers import GoogleCloudEnterpriseSearchRetriever
import vertexai

# init Vertex AI
PROJECT_ID = "<your_project_id>"
REGION = "us-central1"
SEARCH_ENGINE_ID = "<search_engine_id>"
vertexai.init(project=PROJECT_ID, location=REGION)

st.title("Chat Enterprise Search")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Text model instance integrated with LangChain
llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8, verbose=True)
# Enterprise Search retriever
retriever = GoogleCloudEnterpriseSearchRetriever(project_id=PROJECT_ID, search_engine_id=SEARCH_ENGINE_ID, get_extractive_answers=False)
# Create chain to answer questions
#st.session_state.rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
st.session_state.rqa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something here", disabled=("rqa" not in st.session_state)):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # RetrievalQA
    qa = st.session_state.rqa
    response = qa({"question": prompt}) #use "query" for RetrievalQA

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #st.markdown(response["answer"]) #use "result" for RetrievalQA
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response["answer"].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
