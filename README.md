

### 📄 `README.md`

````markdown
# 🧠 RAG-Powered Multi-Agent Q&A (Single File Edition)

This is a one-file Streamlit app that demonstrates Retrieval-Augmented Generation (RAG) with a basic agent framework.

It allows:
- Semantic search over documents
- Answer generation using OpenAI's LLM
- Agent routing to either a calculator, dictionary, or the RAG pipeline

---

## 🚀 How to Run

1. **Create a folder named `data/`** in the same directory as `rag_assistant.py`.
2. **Add a few `.txt` files** into `data/` for ingestion.
3. Install dependencies:

```bash
pip install -r requirements.txt
````

4. Set your OpenAI API key:

```bash
# Linux/macOS
export OPENAI_API_KEY=your_openai_key

# Windows CMD
set OPENAI_API_KEY=your_openai_key
```

5. Run the app:

```bash
streamlit run rag_assistant.py
```

---

  Features

*  Vector search using FAISS and LangChain
*  Natural language response using OpenAI
* ⚙ Simple agent logic:

  * `calculate ...` → routes to calculator
  * `define ...` / `what is ...` → dictionary
  * everything else → RAG + LLM
 *  Streamlit interface with explanation of tool used and sources

---

 Directory

```
main.py         # Single-file implementation
requirements.txt         # Python dependencies
data/                    # Folder containing your .txt documents
```

-

 Example Questions

* `Calculate 10 * (5 + 2)`
* `Define LLM`
* `What does the product documentation say about latency?`

````

---

### 📦 `requirements.txt`

```txt
streamlit
langchain
langchain-community
openai
faiss-cpu
tiktoken
python-dotenv
````


