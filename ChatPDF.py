import tempfile, os, time
import streamlit as st
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import vertexai

# init Vertex AI
# PROJECT_ID = "<your_project_id>"
REGION = "us-central1"
vertexai.init(location=REGION)

st.title("PDF の内容で回答する Chatbot")

if "rqa" not in st.session_state:
    st.session_state.messages = []
    st.write("""
         - [Alphabet 2022 10K annual report](https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf)
         - [Google Cloud セキュリティ ホワイトペーパー](https://services.google.com/fh/files/misc/security_whitepapers_4_booklet_jp.pdf)
         - [「Google Cloud Day: Digital '22 - 15 のトピックから学ぶ」🌟eBook](https://lp.cloudplatformonline.com/rs/808-GJW-314/images/Google_ebooks_all_0614.pdf)
        """)

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
                embedding = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                content = "\n\n".join(doc.page_content for doc in documents)
                texts=text_splitter.split_text(content)
                # Store docs in local vectorstore as index
                db = Chroma.from_texts(texts, embedding)
                # Expose index to the retriever
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                # Text model instance integrated with LangChain
                llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8, verbose=True)
                # Create chain to answer questions
                st.session_state.rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                #st.session_state.rqa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,)
                # bot message
                firstMessage = f"PDF は {len(documents)} ページありました。質問をどうぞ。"
                st.session_state.messages.append({"role": "assistant", "content": firstMessage})

st.divider()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("日本語変換の確定でサブミットされるため、質問はペーストしてください。", disabled=("rqa" not in st.session_state)):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # RetrievalQA
    qa = st.session_state.rqa
    response = qa({"query": prompt}) #"query" for RetrievalQA "question" for ConversationalRetrievalChain

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response["result"].split(): #"result" for RetrievalQA "answer" for ConversationalRetrievalChain
            full_response += chunk + " "
            time.sleep(0.1)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})
