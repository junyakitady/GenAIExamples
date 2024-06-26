import tempfile
import os
import streamlit as st
import langchain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "asia-northeast1"
vertexai.init(project=PROJECT_ID, location=REGION)

st.title("PDF の内容で回答する Chatbot")

if "rqa" not in st.session_state:
    st.session_state.messages = []

# PDF load and embedding
uploaded_file = st.file_uploader("PDF ファイルをアップロードしてください。", type=["pdf"],)
if uploaded_file and ("rqa" not in st.session_state):
    with st.spinner('Loading ...'):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
                # load PDF file
                documents = PyPDFLoader(temp_pdf.name).load()
                # Embeddings API integrated with LangChain
                embedding = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual@latest")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                content = "\n\n".join(doc.page_content for doc in documents)
                texts = text_splitter.split_text(content)
                # Store docs in local vectorstore as index
                db = Chroma.from_texts(texts, embedding)
                # Expose index to the retriever
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                # Text model instance integrated with LangChain
                llm = VertexAI(model_name="gemini-1.5-flash", max_output_tokens=2048, temperature=0.5, verbose=True)
                # Create chain to answer questions
                template = """次のコンテキスト情報を利用して、最後の質問に答えてください。回答は300字程度で回答してください。:
Context: {context}

Question: {question}"""
                prompt = PromptTemplate.from_template(template)
                st.session_state.rqa = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

                # bot message
                firstMessage = f"PDF は {len(documents)} ページありました。質問をどうぞ。"
                st.session_state.messages.append({"role": "assistant", "content": firstMessage})

st.divider()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("日本語変換の確定でサブミットされるため、質問はペーストしてください。", disabled=("rqa" not in st.session_state)):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # RetrievalQA
    qa = st.session_state.rqa
    response = qa.invoke(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})
