import tempfile, os, time
import streamlit as st
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
import vertexai

# init Vertex AI
# PROJECT_ID = "<your_project_id>"
REGION = "us-central1"
vertexai.init(location=REGION)

st.title("Chat PDF with PaLM API")

if "rqa" not in st.session_state:
    st.session_state.messages = []
    st.write("""
         - [Alphabet 2022 10K annual report](https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf)
         - [Google security overview](https://cloud.google.com/static/docs/security/overview/resources/google_security_wp.pdf)
         - [Google infrastructure security design overview](https://cloud.google.com/static/docs/security/infrastructure/design/resources/google_infrastructure_whitepaper_fa.pdf)
        """)

# PDF load and embedding
uploaded_file = st.file_uploader("", type=["pdf"],)
if uploaded_file and ("rqa" not in st.session_state):
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            # load PDF file
            documents = PyPDFLoader(temp_pdf.name).load()
            # Embeddings API integrated with LangChain
            embedding = VertexAIEmbeddings()
            # Store docs in local vectorstore as index
            db = Chroma.from_documents(documents, embedding)
            # Expose index to the retriever
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            # Text model instance integrated with LangChain
            llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8, verbose=True)
            # Create chain to answer questions
            #st.session_state.rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.rqa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,)
            # bot message
            firstMessage = "You are ready to ASK!"
            st.session_state.messages.append({"role": "assistant", "content": firstMessage})

st.divider()
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
