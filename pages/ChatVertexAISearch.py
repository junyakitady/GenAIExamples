from operator import itemgetter
import streamlit as st
import langchain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
    llm = VertexAI(model_name="text-bison", max_output_tokens=512, temperature=0.1, top_k=40, top_p=0.8, verbose=True)
    # Vertex AI Search retriever
    retriever = GoogleVertexAISearchRetriever(project_id=PROJECT_ID, search_engine_id=DATASTORE_ID,
                                              get_extractive_answers=True, max_extractive_answer_count=3, max_documents=3)
    # Create chain to answer questions
    template = """次のコンテキスト情報をもとに、最後の質問に答えてください。
もしコンテキスト情報の中に回答に十分な情報がない場合は、十分な情報がないため分かりません、と答えてください。:
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
