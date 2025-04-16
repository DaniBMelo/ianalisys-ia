# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

st.set_page_config(page_title="IAnalisys - Neuropsicanálise e Autismo", layout="wide")
st.title("🧠 IAnalisys – IA em Neuropsicanálise e Autismo")
st.markdown("""
Esta é uma inteligência artificial treinada com conteúdos específicos sobre autismo e neuropsicanálise.
Faça uma pergunta abaixo para obter respostas baseadas nos textos embarcados.
""")

# ===== Carregar e processar os documentos embarcados =====
@st.cache_resource(show_spinner=True)
def carregar_base():
    docs = []

    # PDF
    loader_pdf = PyPDFLoader("dados/a_trajetoria_autismo_psicanalise.pdf")
    paginas_pdf = loader_pdf.load()

    # DOCX
    loader_docx = UnstructuredWordDocumentLoader("dados/autismo_estrutura_clinica.docx")
    paginas_docx = loader_docx.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs.extend(splitter.split_documents(paginas_pdf))
    docs.extend(splitter.split_documents(paginas_docx))

    # Criar a base vetorial com FAISS
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embedding=embedding)
    return vectorstore

vectorstore = carregar_base()

# ===== Criar o modelo de linguagem =====
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())

# ===== Interface com o usuário =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.text_input("Digite sua pergunta aqui:", key="input_text")

if st.session_state.input_text:
    with st.spinner("Pensando..."):
        resultado = qa_chain({
            "question": st.session_state.input_text,
            "chat_history": st.session_state.chat_history
        })
        resposta = resultado['answer']
        st.session_state.chat_history.append((st.session_state.input_text, resposta))
        st.session_state.input_text = ""  # limpa corretamente sem loop
        st.rerun()

# Exibe o histórico de conversa
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📂 Histórico de perguntas")
    for i, (pergunta, resposta) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**{i+1}.** _{pergunta}_\n> {resposta}")

