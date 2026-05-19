# 🤖 Smart Document Q&A Assistant
### RAG (Retrieval-Augmented Generation) Application
#### Built with Gemini API · ChromaDB · SQLite · Gradio · LangChain
## Recruiter Highlights

- Built an end-to-end AI Document Q&A Assistant using Retrieval-Augmented Generation (RAG)
- Implemented document summarization and context-aware question answering
- Used embeddings and ChromaDB for semantic search and document retrieval
- Integrated Gemini API with LangChain for LLM-based responses
- Added SQLite support for storing and managing document/query data
- Created an interactive Gradio interface for easy user interaction
- Designed the project to solve a real-world problem: quickly understanding long documents.

---

## 📌 Table of Contents
1. [Project Overview](#overview)
2. [Architecture Diagram](#architecture)
3. [Tech Stack](#tech-stack)
4. [Folder Structure](#folder-structure)
5. [Setup & Installation](#setup)
6. [Running in Google Colab](#colab)
7. [Configuration Reference](#config)
8. [Pipeline Deep Dive](#pipeline)
9. [SQLite Schema Reference](#sqlite)
10. [Gradio UI Guide](#ui)
11. [Sample Test Questions](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Evaluation Metrics](#evaluation)
14. [Future Improvements](#future)

---

## 📋 Project Overview <a name="overview"></a>

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** that allows users to upload documents (PDF or TXT), have them automatically chunked and indexed into a vector database, and then ask natural language questions — receiving answers grounded in the actual document content.

### Core Capabilities

| Capability | Details |
|-----------|---------|
| Document Ingestion | PDF and TXT files via Colab upload or Gradio UI |
| Text Chunking | Recursive Character Splitting (LangChain) |
| Semantic Embeddings | Google Gemini `text-embedding-004` (768-dim) |
| Vector Storage | ChromaDB with cosine similarity indexing |
| RAG Retrieval | Top-K semantic search + context injection |
| Answer Generation | Gemini 1.5 Flash with structured RAG prompt |
| Query Logging | SQLite with full schema (query, chunks, latency) |
| Frontend | Gradio Blocks with 4-tab dark-theme UI |

---

## 🏗️ Architecture Diagram <a name="architecture"></a>

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                              │
│  User Upload (PDF/TXT)                                       │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────┐    ┌───────────────────────┐               │
│  │   PyMuPDF   │    │  RecursiveCharacter   │               │
│  │  Text Ext.  │───►│  TextSplitter         │               │
│  │  (fitz)     │    │  (chunk=800,ovlp=150) │               │
│  └─────────────┘    └───────────┬───────────┘               │
│                                 │                            │
│                                 ▼                            │
│                    ┌────────────────────────┐               │
│                    │  Gemini Embedding API   │               │
│                    │  text-embedding-004     │               │
│                    │  (768-dim vectors)      │               │
│                    └───────────┬────────────┘               │
│                                │                            │
│                                ▼                            │
│                    ┌────────────────────────┐               │
│                    │      ChromaDB          │               │
│                    │  (Persistent, Cosine)  │               │
│                    └────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                              │
│  User Query (Gradio Chat)                                    │
│        │                                                     │
│        ▼                                                     │
│  Gemini Embed Query ──► ChromaDB Top-K Search               │
│                                 │                            │
│                    Top-K Relevant Chunks                     │
│                                 │                            │
│                                 ▼                            │
│              ┌──────────────────────────────┐               │
│              │     RAG Prompt Builder        │               │
│              │  [System Prompt]              │               │
│              │  [Context 1..K]               │               │
│              │  [User Question]              │               │
│              └──────────────┬───────────────┘               │
│                             │                                │
│                             ▼                                │
│              ┌──────────────────────────────┐               │
│              │    Gemini 1.5 Flash LLM       │               │
│              │    (temp=0.3, max_tok=1024)   │               │
│              └──────────────┬───────────────┘               │
│                             │                                │
│              ┌──────────────┴───────────────┐               │
│              │              │               │               │
│              ▼              ▼               ▼               │
│         SQLite Log     Gradio UI      Evidence Panel        │
│         (query_logs)   (Answer)       (Source Chunks)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack <a name="tech-stack"></a>

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| LLM | `google-generativeai` | ≥0.7 | Gemini 1.5 Flash for answer generation |
| Embeddings | `google-generativeai` | ≥0.7 | Gemini text-embedding-004 |
| Vector DB | `chromadb` | ≥0.5 | Store & query chunk embeddings |
| Text Splitting | `langchain-text-splitters` | ≥0.2 | RecursiveCharacterTextSplitter |
| PDF Parsing | `pymupdf` (fitz) | ≥1.23 | Extract text from PDF pages |
| Chat UI | `gradio` | ≥4.0 | Gradio Blocks interface |
| Query Logging | `sqlite3` | stdlib | Built-in Python SQLite driver |
| Runtime | Google Colab | — | Free GPU/CPU cloud notebook |

---

## 📁 Folder Structure <a name="folder-structure"></a>

```
project/
├── Smart_Document_QA_Assistant.ipynb   ← Main notebook (all-in-one)
├── requirements.txt                    ← pip install reference
├── README.md                          ← This file
└── (auto-generated at runtime)
    ├── /content/chroma_db/            ← ChromaDB persistent store
    ├── /content/query_logs.db         ← SQLite database
    └── /content/AI_Overview_Sample.txt ← Built-in test document
```

---

## ⚙️ Setup & Installation <a name="setup"></a>

### Prerequisites
- Google Account (for Colab)
- Gemini API Key — free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Install Dependencies (Cell 1)
```bash
pip install google-generativeai chromadb gradio pymupdf \
            langchain langchain-community langchain-text-splitters python-dotenv
```

---

## ▶️ Running in Google Colab <a name="colab"></a>

### Step-by-Step

```
1. Open Google Colab: https://colab.research.google.com
2. File → Upload notebook → Select Smart_Document_QA_Assistant.ipynb
3. Runtime → Run all   (or Ctrl+F9)
4. When Cell 3 prompts: enter your Gemini API key (hidden input)
5. Wait ~2 minutes for all cells to complete
6. Cell 13 prints a public URL: https://xxxx.gradio.live
7. Open the URL — your app is live!
```

### Recommended Runtime
- **CPU** is sufficient (no GPU needed)
- Free Colab tier works fine

### Important Notes
- If prompted to "Restart Runtime" after Cell 1, click it — then re-run from Cell 2
- The public Gradio link expires after 72 hours
- ChromaDB data persists only for the Colab session

---

## 🔧 Configuration Reference <a name="config"></a>

All tunable parameters are in **Cell 4** (`Section 4 — Global Configuration`):

| Constant | Default | Effect |
|----------|---------|--------|
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `LLM_MODEL` | `gemini-1.5-flash` | Answer generation model |
| `CHUNK_SIZE` | `800` | Characters per chunk (smaller = more precise, slower) |
| `CHUNK_OVERLAP` | `150` | Shared chars between adjacent chunks (prevents boundary loss) |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query (higher = more context, more tokens) |
| `CHROMA_COLLECTION` | `document_store` | ChromaDB collection name |
| `CHROMA_DB_PATH` | `/content/chroma_db` | Persistence path |
| `SQLITE_DB_PATH` | `/content/query_logs.db` | SQLite file path |

### Tuning Tips

**For higher accuracy:**
- Decrease `CHUNK_SIZE` to 400–600 (more granular chunks)
- Increase `TOP_K_RESULTS` to 7–10

**For faster responses:**
- Increase `CHUNK_SIZE` to 1000–1200
- Decrease `TOP_K_RESULTS` to 3

**For longer documents:**
- Increase `CHUNK_SIZE` to 1000
- Keep overlap at ~15–20% of chunk size

---

## 🔬 Pipeline Deep Dive <a name="pipeline"></a>

### 1. Text Extraction

**PDF** — PyMuPDF (`fitz`) extracts text page-by-page, inserts `[Page N]` markers, and collapses excessive newlines.

**TXT** — UTF-8 first, falls back to `latin-1` then `cp1252` for legacy files.

### 2. Recursive Character Chunking

LangChain's `RecursiveCharacterTextSplitter` splits on this hierarchy:
```
\n\n  →  \n  →  ". "  →  " "  →  ""
```
It tries the first separator; if chunks are still too large, it recurses to the next. This preserves paragraph and sentence boundaries wherever possible.

**Why not fixed-size chunking?**
Fixed-size splits often cut mid-sentence, losing semantic context. Recursive splitting respects natural language structure.

### 3. Gemini Embeddings

- Uses `text-embedding-004` — Google's latest general-purpose embedding model
- **768-dimensional** vectors
- Two task types used:
  - `retrieval_document` — when storing chunks (optimised for being found)
  - `retrieval_query` — when embedding a user query (optimised for finding)
- Batched in groups of 10 with 300ms delay to respect rate limits
- Exponential backoff retry (1s → 2s → 4s) on failures

### 4. ChromaDB Vector Store

- **Cosine similarity** (`hnsw:space: cosine`) — distance 0 = identical, 1 = orthogonal
- `upsert()` used instead of `add()` — safe to re-run without duplicates
- Stored metadata per chunk: `source` filename + `chunk_idx`

### 5. RAG Prompt Engineering

The prompt passed to Gemini follows this structure:
```
[System Instructions]     ← Grounding rules, citation guidelines
[Context 1..K]            ← Retrieved chunks with source + distance labels
[User Question]           ← Verbatim user input
[Answer:]                 ← LLM continues from here
```
Temperature is set to **0.3** — low enough for factual grounding, high enough for natural phrasing.

---

## 🗄️ SQLite Schema Reference <a name="sqlite"></a>

### Table: `query_logs`
```sql
CREATE TABLE query_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,   -- Groups a chat session
    timestamp        TEXT    NOT NULL,   -- ISO-8601 UTC
    user_query       TEXT    NOT NULL,
    retrieved_chunks TEXT,               -- JSON array of chunk texts
    llm_answer       TEXT,
    source_docs      TEXT,               -- Comma-separated filenames
    latency_ms       REAL                -- End-to-end time in ms
);
```

### Table: `document_registry`
```sql
CREATE TABLE document_registry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    UNIQUE NOT NULL,
    file_hash    TEXT    NOT NULL,       -- MD5 for deduplication
    chunk_count  INTEGER NOT NULL,
    uploaded_at  TEXT    NOT NULL        -- ISO-8601 UTC
);
```

### Useful Analytics Queries
```sql
-- Average response latency
SELECT AVG(latency_ms) AS avg_ms FROM query_logs;

-- Most queried sources
SELECT source_docs, COUNT(*) AS queries
FROM query_logs GROUP BY source_docs ORDER BY queries DESC;

-- Queries slower than 5 seconds
SELECT user_query, latency_ms FROM query_logs WHERE latency_ms > 5000;

-- Total chunks per document
SELECT filename, chunk_count FROM document_registry;
```

---

## 🎨 Gradio UI Guide <a name="ui"></a>

The interface has **4 tabs**:

### Tab 1: 💬 Chat
| Element | Description |
|---------|-------------|
| Upload panel | Drag-drop PDF/TXT files, click "Process Document(s)" |
| Document list | Shows all ingested files |
| Chat window | Type your question, press Enter or click Send |
| Evidence panel | Shows top-K retrieved chunks with source + distance |

### Tab 2: 📊 Query Logs
- Slider to select how many recent logs to show
- Table view: timestamp, query, source doc, latency

### Tab 3: 🗄️ DB Schema
- Full schema documentation for both SQLite tables
- Example SQL queries

### Tab 4: ⚙️ System Info
- All active configuration values
- Session ID for cross-referencing logs

---

## 🧪 Sample Test Questions <a name="testing"></a>

Use these with the pre-loaded sample document (`AI_Overview_Sample.txt`):

```
1. What is Retrieval-Augmented Generation and how does it help with hallucinations?
2. Who coined the term "Artificial Intelligence" and in what year?
3. What is the difference between supervised, unsupervised, and reinforcement learning?
4. What are the ethical concerns around AI development?
5. How do Convolutional Neural Networks differ from Recurrent Neural Networks?
6. What economic impact is AI projected to have by 2030?
7. What is the difference between Narrow AI and General AI?
8. Explain how the Transformer model changed Natural Language Processing.
```

---

## 🔧 Troubleshooting <a name="troubleshooting"></a>

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError` | Cell 1 not run / runtime not restarted | Run Cell 1, restart runtime, re-run from Cell 2 |
| `Invalid API key` | Wrong key entered | Get a fresh key from aistudio.google.com |
| `No documents in vector store` | Ingestion failed silently | Re-run Cell 10 (sample doc) or re-upload |
| `Embedding failed after 3 attempts` | Gemini API rate limit | Wait 60s, reduce batch size in Cell 6 |
| `Gradio link not working` | Session expired | Re-run Cell 13 |
| PDF returns empty text | Image-only / scanned PDF | Use a text-based PDF or convert with OCR first |
| ChromaDB `InvalidDimensionException` | Changed embedding model mid-session | Run `reset_collection()` in a new cell |

---

## 📈 Evaluation Metrics <a name="evaluation"></a>

After running queries, evaluate the system using these criteria:

### Retrieval Quality
- **Relevance Score**: Average cosine distance of retrieved chunks (lower = better match)
- **Source Diversity**: Number of distinct source documents per query

### Generation Quality
- **Groundedness**: Does the answer use only information from the context?
- **Completeness**: Does it fully address the question?
- **Latency**: Target < 5000ms end-to-end

### SQLite-based Monitoring
```python
import pandas as pd, sqlite3
conn = sqlite3.connect("/content/query_logs.db")
df = pd.read_sql("SELECT * FROM query_logs", conn)
print(df[["user_query","latency_ms"]].describe())
```

---

## 🚀 Future Improvements <a name="future"></a>

| Feature | Implementation Idea |
|---------|-------------------|
| **Multi-modal** | Add image support via Gemini Vision |
| **Conversation memory** | Maintain rolling chat history in the RAG prompt |
| **Re-ranking** | Add a cross-encoder re-ranker after initial retrieval |
| **Hybrid search** | Combine BM25 keyword search + vector search |
| **Streaming** | Stream Gemini responses token-by-token in Gradio |
| **Auth** | Add Gradio auth for multi-user deployments |
| **Export** | Export Q&A sessions as PDF report |
| **Evaluation** | Integrate RAGAS framework for automated RAG evaluation |
| **Cloud Deploy** | Deploy to Hugging Face Spaces or Google Cloud Run |

---

## 📄 License
MIT — free for educational and commercial use.

---

*Built for AI Assessment · Powered by Google Gemini + ChromaDB + Gradio + LangChain*
