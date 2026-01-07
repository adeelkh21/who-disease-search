# WHO Disease Semantic Search System

> A production-ready vector database for searching World Health Organization disease fact sheets using advanced NLP embeddings and semantic similarity

## 📖 Overview

This project implements an end-to-end semantic search system that transforms **1,652 WHO disease fact sheets** into a searchable vector database. It combines web scraping, intelligent text preprocessing, deep learning embeddings, and vector indexing to enable fast, accurate medical information retrieval.

The system supports multiple search modalities including pure vector similarity, hybrid search with keyword matching (BM25), and cross-encoder reranking for improved result quality.

## 🏗️ Architecture & Pipeline

### Data Pipeline
```
WHO Website → Scraping → Text Extraction → Cleaning → Chunking → Embeddings → Vector Index
```

### Key Components

1. **Web Scraping** (`WHO_Disease_Scrapper.py`)
   - Extracts 233 WHO fact sheet pages
   - 100% success rate with polite 1-second request delays
   - Parses structured HTML to recover semantic organization

2. **Data Processing & Chunking**
   - Removes citations, normalizes whitespace, cleans noise
   - Intelligent sentence-aware chunking (200-450 tokens per chunk)
   - Smart merging of small chunks to maintain quality
   - Results: **1,652 high-quality text chunks**

3. **Embedding Generation** (`generate_embeddings.py`)
   - Uses **Bio_ClinicalBERT** (768-dimensional embeddings)
   - Purpose-built for biomedical text understanding
   - Preserves semantic meaning of medical terminology

4. **Vector Indexing** (`build_index.py`)
   - **FAISS IndexFlatL2** for fast similarity search
   - L2 distance metric on normalized vectors (cosine equivalence)
   - Sub-100ms query latency

5. **Search & Retrieval**
   - **Vector Search**: Fast semantic similarity (`search_index.py`)
   - **Hybrid Search**: Combines vectors + BM25 keyword matching (`search_enhanced.py`)
   - **Reranking**: Cross-encoder reranking for result refinement

## 📊 Dataset Details

| Metric | Value |
|--------|-------|
| **Data Source** | WHO Fact Sheets (https://www.who.int/news-room/fact-sheets) |
| **Total Chunks** | 1,652 |
| **Pages Scraped** | 233 |
| **Average Chunk Size** | 180.85 tokens |
| **Format** | JSONL (JSON Lines) |
| **Key Fields** | `id`, `title`, `section`, `text`, `source_url`, `category` |


## ⚡ Quick Start

#### Install dependencies
pip install -r requirements_embeddings.txt


### Running from Scratch

#### 1. Scrape WHO data
python WHO_Disease_Scrapper.py

#### 2. Generate embeddings
python generate_embeddings.py

#### 3. Build FAISS index
python build_index.py

#### 4. Test the system
python demo.py


## 🔍 Search Capabilities

| Feature | Method | Speed | Quality | Use Case |
|---------|--------|-------|---------|----------|
| **Pure Vector** | `search_index.py` | ⚡ Fast | ⭐⭐⭐ Good | Real-time queries |
| **Hybrid** | `--hybrid` flag | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | Keyword + semantic |
| **Reranked** | `--rerank` flag | ⚡⚡⚡ Slower | ⭐⭐⭐⭐⭐ Best | Production ranking |


## 🎯 Key Technologies

- **Language Model**: Bio_ClinicalBERT (HuggingFace)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **NLP**: spaCy, NLTK, tiktoken
- **Search**: BM25 (rank_bm25), Cross-encoder reranking
- **Data Format**: JSONL for streaming efficiency

## 📈 Performance Metrics

- **Query Latency**: 57-87ms per query
- **Index Size**: 1,652 vectors × 768 dimensions
- **Relevance**: ~50% for complex biomedical queries
- **Scraping Success**: 100% (233/233 pages)
- **Data Quality**: Citation filtering, sentence-aware chunking


## 🎓 Learning Outcomes

This project demonstrates:
- Large-scale web scraping and data engineering
- NLP preprocessing and text normalization  
- Semantic embeddings and vector databases
- Similarity search and ranking algorithms
- Production-ready system design
- Performance evaluation and metrics
