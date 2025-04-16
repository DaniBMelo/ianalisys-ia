# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.embeddings import OpenAIEmbeddings

#Lê a chave da OpenAI dos segredos do Streamlit
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
    loader_pdf = PyPDFLoader("dados/A_TRAJETORIA_HISTORICO-CONCEITUAL_DO_AUTISMO_NA_PSICANALISE.pdf")
    paginas_pdf = loader_pdf.load()

    # DOCX
    loader_docx = UnstructuredWordDocumentLoader("dados/Autismo-A_Questao_Estrutural_e_Suas_Implicacoes_na_Clinica.docx")
    paginas_docx = loader_docx.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs.extend(splitter.split_documents(paginas_pdf))
    docs.extend(splitter.split_documents(paginas_docx))

    # Criar a base vetorial com Chroma
    vectorstore = FAISS.from_documents(docs, embedding=embedding)

    return vectorstore

vectorstore = carregar_base()

# ===== Criar o modelo de linguagem =====
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())

# ===== Interface com o usuário =====

# Inicializa histórico e controle da última pergunta
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Campo de entrada de pergunta
user_input = st.text_input("Digite sua pergunta aqui:", key="input_pergunta")

# Processa nova pergunta (evita repetir após rerun)
if user_input and user_input != st.session_state.last_question:
    with st.spinner("Pensando..."):
        resultado = qa_chain({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        resposta = resultado['answer']
        st.session_state.chat_history.append((user_input, resposta))
        st.session_state.last_question = user_input
        st.rerun()  # limpa o campo e reinicia a interface

# Exibe o histórico de conversa
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🗂️ Histórico de perguntas")
    for i, (pergunta, resposta) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**{i+1}.** _{pergunta}_\n> {resposta}")
