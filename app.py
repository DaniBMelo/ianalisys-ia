# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
import os

st.set_page_config(page_title="IAnalysis - IA em Neuropsicanálise", layout="wide")
st.title("🧠 Ianalysis – IA em Neuropsicanálise e Autismo")

st.markdown("""
Esta é uma inteligência artificial criada por Danila Melo, treinada com conteúdos específicos sobre neuropsicanálise e autismo.
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

# ===== Controle de tokens acumulados na sessão =====
if "tokens_acumulados" not in st.session_state:
    st.session_state.tokens_acumulados = 0
if "custo_acumulado" not in st.session_state:
    st.session_state.custo_acumulado = 0.0

# ===== Interface com o usuário =====
if "historico_projetos" not in st.session_state:
    st.session_state.historico_projetos = {"Geral": []}  # começa com um projeto padrão

# Lista de projetos (você pode expandir depois)
projetos = ["Geral", "TEA adultos", "Neurodesenvolvimento", "Supervisão", "Outros"]

# ----- Cadastro de novo projeto -----
st.markdown("### ➕ Criar novo projeto")
novo_projeto = st.text_input("Nome do novo projeto:", key="novo_projeto_input")

if st.session_state.get("limpar_projeto_input"):
    st.session_state["novo_projeto_input"] = ""
    st.session_state["limpar_projeto_input"] = False


if st.session_state.get("limpar_projeto_input"):
    st.session_state["novo_projeto_input"] = ""
    st.session_state["limpar_projeto_input"] = False
    
if st.button("Adicionar projeto") and novo_projeto:
    if "historico_projetos" not in st.session_state:
        st.session_state.historico_projetos = {}

    if novo_projeto not in st.session_state.historico_projetos:
        st.session_state.historico_projetos[novo_projeto] = []
        st.session_state.projeto_atual = novo_projeto  # seleciona o novo automaticamente
        st.session_state["limpar_projeto_input"] = True  # define a flag para limpar depois
        st.success(f"✅ Projeto '{novo_projeto}' adicionado com sucesso!")
    else:
        st.warning("⚠️ Esse projeto já existe.")

projetos = list(st.session_state.historico_projetos.keys())
projeto_atual = st.selectbox("📂 Selecione um projeto:", projetos, key="projeto_atual")

# Cria histórico para cada projeto se ainda não existir
if "historico_projetos" not in st.session_state:
    st.session_state.historico_projetos = {p: [] for p in projetos}

# ----- Upload de arquivos -----
st.markdown("### 📂 Adicionar documento ao projeto atual")
uploaded_file = st.file_uploader("Escolha um arquivo PDF ou DOCX", type=["pdf", "docx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    with st.spinner("Processando arquivo..."):
        import tempfile
        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Carrega com o loader apropriado
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_type == "docx":
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
        else:
            st.error("❌ Formato não suportado.")
            st.stop()

        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        novo_vectorstore = FAISS.from_documents(docs, embedding=embedding)
        qa_chain.retriever.vectorstore.merge_from(novo_vectorstore)
        st.success("✅ Documento adicionado com sucesso à base de conhecimento!")

if "executar" not in st.session_state:
    st.session_state.executar = False
if "pergunta_temp" not in st.session_state:
    st.session_state.pergunta_temp = ""

# Função chamada ao enviar
def enviar():
    st.session_state.executar = True
    st.session_state.pergunta_temp = st.session_state.get("input_text", "")
    st.session_state["input_text"] = ""  # limpa o campo após enviar

# Campo de entrada (sem sobrescrever o valor)
st.text_input(
    "Digite sua pergunta aqui:",
    value=st.session_state.get("input_text", ""),
    key="input_text",
    on_change=enviar
)

# Processa a pergunta apenas se a flag for verdadeira
if st.session_state.executar and st.session_state.pergunta_temp:
    with st.spinner("Pensando..."):
        with get_openai_callback() as cb:
            resultado = qa_chain({
                "question": st.session_state.pergunta_temp,
                "chat_history": st.session_state.historico_projetos[projeto_atual]
            })
            resposta = resultado['answer']
        
            # Atualiza os acumuladores
            st.session_state.tokens_acumulados += cb.total_tokens
        
            # Estima custo com base no modelo
            # (ajuste os valores se usar outro modelo)
            custo_input = cb.prompt_tokens * 0.0005 / 1000
            custo_output = cb.completion_tokens * 0.0015 / 1000
            custo_total = custo_input + custo_output
            st.session_state.custo_acumulado += custo_total

            st.markdown(f"🔢 **Tokens usados nesta resposta:** {cb.total_tokens}")


    st.session_state.historico_projetos[projeto_atual].append((st.session_state.pergunta_temp, resposta))

    # Limpa apenas as variáveis de controle (não sobrescreve o input controlado)
    st.session_state.executar = False
    st.session_state.pergunta_temp = ""

# Exibe o histórico de conversa
if st.session_state.historico_projetos.get(projeto_atual):
    st.markdown("---")
    st.markdown("### 📂 Histórico de perguntas")
    for i, (pergunta, resposta) in enumerate(reversed(st.session_state.historico_projetos[projeto_atual])):
        st.markdown(f"**{i+1}.** _{pergunta}_\n> {resposta}")


with st.sidebar:
    st.markdown("## 💰 Monitor de uso")
    st.markdown(f"**Modelo:** gpt-3.5-turbo")
    st.markdown(f"**Tokens acumulados:** {st.session_state.tokens_acumulados}")
    st.markdown(f"**Custo estimado:** ${st.session_state.custo_acumulado:.4f}")
