import tempfile
import os
import streamlit as st
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import vertexai

st.title("Vertex AI による生成 AI サンプル集")

st.link_button("Chat PDF", "./ChatPDF")
st.link_button("Chat Vertex AI Search", "./ChatVertexAISearch")
st.link_button("Compare Gemini and PaLM", "./Gemini_vs_PaLM")
st.link_button("Ask BigQuery", "./Ask_BigQuery")
