import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA

# -------------------- SETUP --------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    convert_system_message_to_human=True
)

# Gemini Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = "langchainpdf"  # your existing index name

# ------------------ Corrective RAG Function ------------------
def corrective_rag_query(query: str, qa_chain):
    st.write(f"🧠 Query: {query}")
    
    first_response = llm.invoke(query)

    if isinstance(first_response, list):
        first_response = first_response[0]

    try:
        first_response_text = first_response.content.strip()
    except AttributeError:
        first_response_text = str(first_response).strip()

    st.markdown("🔹 **Initial Gemini Response**")
    st.info(first_response_text)

    if "I don't know" in first_response_text or "I'm not sure" in first_response_text or len(first_response_text) < 50:
        st.warning("⚠️ Low confidence — using RAG for better result...")
        improved_response = qa_chain.invoke(query)
        raw_answer = str(improved_response["result"])

        match = re.search(r"content=\[(.*?)\]", raw_answer, re.DOTALL)
        if match:
            try:
                content_list = eval(f"[{match.group(1)}]")
                cleaned_answer = " ".join(s.strip() for s in content_list)
            except Exception as e:
                cleaned_answer = "⚠️ Error parsing content: " + str(e)
        else:
            cleaned_answer = raw_answer.strip()

        return {"result": cleaned_answer}
    
    return {"result": first_response_text}

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="RAG App with Gemini + Pinecone", layout="wide")
st.title("📄🔍 RAG PDF QA App using Gemini + Pinecone")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

# Namespace input
namespace = st.text_input("🧾 Enter namespace (lowercase, no space):", "llmpdf")

if uploaded_file is not None and st.button("🚀 Process PDF and Create Vector DB"):
    with st.spinner("Processing PDF..."):
        # Save uploaded PDF temporarily
        file_path = os.path.join("documents", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        vectorstore = LangchainPinecone.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=INDEX_NAME,
            namespace=namespace
        )

        st.success("✅ PDF processed and embeddings stored in Pinecone")

        # Save retriever & QA chain globally
        st.session_state.retriever = vectorstore.as_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.retriever,
            return_source_documents=True
        )

# -------------------- Ask Questions --------------------
st.markdown("---")
st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question here:")

if st.button("🎯 Get Answer") and query:
    if "qa_chain" not in st.session_state:
        st.error("⚠️ Please upload and process a PDF first.")
    else:
        result = corrective_rag_query(query, st.session_state.qa_chain)
        raw_answer = result["result"]

        # Optional cleanup (in case it's in content=[...] format)
        match = re.search(r"content=\[(.*?)\]", raw_answer, re.DOTALL)
        if match:
            try:
                content_list = eval(f"[{match.group(1)}]")
                cleaned_answer = " ".join(s.strip() for s in content_list)
            except:
                cleaned_answer = raw_answer
        else:
            cleaned_answer = raw_answer

        st.markdown("### ✅ Final Answer")
        st.success(cleaned_answer)