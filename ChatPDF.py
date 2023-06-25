import streamlit as st
from streamlit_chat import message
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
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
    st.session_state.user = []
    st.session_state.bot = []
    st.subheader("1. Set a PDF URL to load")
    st.text("- Alphabet 2022 10K annual report")
    st.code("https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf")
    st.text("- Google infrastructure security design overview")
    st.code("https://cloud.google.com/static/docs/security/infrastructure/design/resources/google_infrastructure_whitepaper_fa.pdf")
    st.caption("PyPDF で日本語 PDF のパースがうまく行かない場合があります")


# PDF load and embedding
pdf_url = st.text_input("Enter PDF URL", disabled="rqa" in st.session_state)

if pdf_url and ("rqa" not in st.session_state):
    # Ingest PDF files
    documents = PyPDFLoader(pdf_url).load()
    #for doc in documents:
    #    doc.page_content = doc.page_content.replace('\n', ' ') # remove '\n' in some cases
    # split the documents into smaller chunks (1 page is a good chunk)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    # texts = text_splitter.split_documents(documents)

    # Embeddings API integrated with LangChain
    embedding = VertexAIEmbeddings()
    # Store docs in local vectorstore as index
    db = Chroma.from_documents(documents, embedding)
    # Expose index to the retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Text model instance integrated with LangChain
    llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8, verbose=True)
    # Create chain to answer questions
    st.session_state.rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

if pdf_url and ("rqa" in st.session_state):
    st.text("Please reload this page to reset PDF")
    st.subheader("2. You are ready to ASK!")

st.divider()
message("Hello, load your PDF first.", key="hello")
for i in range(len(st.session_state.bot)):
    message(st.session_state.user[i], is_user=True, key=f"user{i}")
    message(st.session_state.bot[i], key=f"bot{i}")


def on_input_change():
    query = st.session_state.query
    qa = st.session_state.rqa
    # Uses LLM to synthesize results from the search index.
    response = qa({"query": query})
    # render chat
    st.session_state.user.append(query)
    st.session_state.bot.append(response["result"])
    st.session_state.response = response


query = st.text_input("Ask something here", disabled="rqa" not in st.session_state, on_change=on_input_change, key="query",
                      placeholder="Describe how Google protect customer data", )
if "response" in st.session_state:
    st.divider()
    st.caption("Check source documents")
    st.caption(st.session_state.response["source_documents"])
