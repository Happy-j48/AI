import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# --- Constants ---
VECTOR_DB_PATH = "vector_store"

# --- Tool: Calculator ---
def calculate_expression(expr):
    try:
        result = eval(expr.replace("calculate", ""))
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

# --- Tool: Dictionary ---
def define_term(term):
    dictionary = {
        "api": "A set of functions allowing applications to access data or features.",
        "llm": "Large Language Model, a type of AI trained on vast text corpora.",
        "rag": "Retrieval-Augmented Generation, combining retrieval with text generation."
    }
    for k in dictionary:
        if k in term.lower():
            return dictionary[k]
    return "Definition not found."

# --- Vector Store Management ---
def ingest_docs():
    docs = []
    for fname in os.listdir("data"):
        if fname.endswith(".txt"):
            loader = TextLoader(f"data/{fname}")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

def init_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        ingest_docs()

def get_relevant_chunks(query):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    results = db.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

# --- Agent Logic ---
llm = ChatOpenAI(temperature=0)
qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

def agent_router(query):
    q = query.lower()

    if any(kw in q for kw in ["calculate", "add", "subtract", "*", "/", "+", "-"]):
        result = calculate_expression(query)
        return "calculator", None, result

    if "define" in q or "what is" in q:
        result = define_term(query)
        return "dictionary", None, result

    context = get_relevant_chunks(query)
    answer = qa_chain.run({"input_documents": context, "question": query})
    return "RAG+LLM", context, answer

# --- Streamlit UI ---
st.title("RAG-Powered Multi-Agent Q&A Assistant")

query = st.text_input("Ask a question:")

if query:
    decision, context, answer = agent_router(query)

    st.markdown(f"**Agent Decision:** `{decision}`")
    if context:
        st.markdown("**Retrieved Context:**")
        for chunk in context:
            st.code(chunk)
    st.markdown("**Answer:**")
    st.success(answer)

init_vector_store()
