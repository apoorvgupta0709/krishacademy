# 🧠 RAG Pipeline with LangChain, MongoDB, and OpenAI

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:
- 📄 **PDF document ingestion**
- 🧠 **Semantic chunking & embedding**
- 🌐 **MongoDB Atlas as a vector store**
- 🤖 **OpenAI LLM for intelligent Q&A**
  
All orchestrated with **LangChain**.

---

## 🚀 Project Flow

```mermaid
flowchart TD
    A[📄 Load PDF] --> B[🔗 Semantic Chunking]
    B --> C[🧬 Generate Embeddings]
    C --> D[🛢️ Store in MongoDB Atlas]
    E[❓ User Query] --> F[🔍 Similarity Search (Retriever)]
    D --> F
    F --> G[🧠 Combine with Prompt Template]
    G --> H[🤖 Query OpenAI LLM]
    H --> I[✅ Final Answer + Source Docs]
```

---

## 🧩 Components

### 1. **Document Ingestion and Chunking**
```python
loader = PyPDFLoader("your_file.pdf")
documents = loader.load()

chunker = SemanticChunker(OpenAIEmbeddings())
chunks = chunker.split_documents(documents)
```
- Loads a PDF and semantically splits it into meaningful chunks.
- Semantic chunking ensures context-rich divisions, better than naive splitting.

---

### 2. **Vector Storage in MongoDB Atlas**
```python
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="default"
)

vector_store.add_documents(chunks)
```
- Embeds each chunk and stores it as a vector in MongoDB Atlas.
- Enables fast semantic search later via MongoDB's vector index.

---

### 3. **Prompt Template + LLM Definition**
```python
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following document content to answer the question.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")
```
- A custom prompt instructs the LLM to **only answer based on retrieved content**.
- If not found in context, it avoids hallucination.

---

### 4. **RAG Chain Setup**
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)
```
- Combines retrieval + prompt + LLM in one callable pipeline.
- Retrieves top-4 semantically similar chunks from MongoDB.

---

### 5. **Query Execution**
```python
query = "What are the key recommendations in the report?"
response = rag_chain.invoke({"query": query})

print("Answer:", response["result"])
print("Sources:", [doc.metadata for doc in response["source_documents"]])
```
- Ask any natural-language question related to the uploaded document.
- Returns both answer **and** source documents used to generate it.

---

## 🛠️ Prerequisites

Install dependencies:

```bash
pip install langchain langchain-openai langchain-experimental pymongo
```

Ensure your MongoDB Atlas instance has:
- A **vectorSearch index** on the appropriate collection
- IP whitelisting for your current machine
- Correct connection URI with credentials

---

## ✅ Example MongoDB Index Config

```json
{
  "fields": [
    {
      "path": "embedding",
      "type": "vector",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ],
  "name": "default",
  "type": "vectorSearch"
}
```

---

## 📂 Folder Structure (recommended)

```
.
├── your_file.pdf
├── main.py
├── README.md
└── requirements.txt
```

---

## 🤝 Contributing

Pull requests welcome! Please format your code with Black and test before submitting.

---

## 📬 Contact

For help, open an issue or connect with [@apoorvgupta0709](https://github.com/apoorvgupta0709)