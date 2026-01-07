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

### Data Record Structure
```json
{
  "id": "disease_name_section_chunk_N",
  "title": "Disease Name",
  "section": "Section Heading",
  "chunk_id": 1,
  "text": "Cleaned medical text content...",
  "source_url": "https://www.who.int/...",
  "source": "WHO",
  "category": "Breadcrumb navigation"
}
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM (minimum)

### Installation

```bash
# Clone and setup
cd who-disease-search
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_embeddings.txt
```

### Running from Scratch

```bash
# 1. Scrape WHO data
python WHO_Disease_Scrapper.py

# 2. Generate embeddings
python generate_embeddings.py

# 3. Build FAISS index
python build_index.py

# 4. Test the system
python demo.py
```

### Search Examples

**Interactive Vector Search:**
```bash
python search_index.py
# Type queries: "What are diabetes symptoms?"
```

**Command-line Search:**
```bash
python search_index.py --query "treatment for malaria"
python search_enhanced.py --query "HIV prevention" --hybrid --rerank
```

**Batch Demo:**
```bash
python demo.py --full  # Runs comprehensive demo with evaluation
```

## 📁 Project Structure

```
.
├── WHO_Disease_Scrapper.py      # Web scraper
├── generate_embeddings.py       # Embedding generation
├── build_index.py               # Index construction
├── search_index.py              # Basic vector search
├── search_enhanced.py           # Hybrid + reranking search
├── demo.py                      # System demonstration
├── evaluation.py                # Performance metrics
├── config/
│   ├── config.py                # Central configuration
│   └── __init__.py
├── utils/
│   ├── embeddings.py            # Embedding utilities
│   ├── io_utils.py              # File I/O operations
│   ├── search_utils.py          # Search operations
│   ├── hybrid_search.py         # BM25 hybrid search
│   └── reranker.py              # Cross-encoder reranking
├── data/
│   ├── who_diseases_Full_Catalogue.jsonl  # Original data
│   ├── who_embeddings.jsonl               # Generated embeddings
│   ├── who_index.faiss                    # Vector index
│   └── index_metadata.json                # Index metadata
├── requirements_embeddings.txt  # Python dependencies
└── README.md                    # This file
```

## 🔍 Search Capabilities

| Feature | Method | Speed | Quality | Use Case |
|---------|--------|-------|---------|----------|
| **Pure Vector** | `search_index.py` | ⚡ Fast | ⭐⭐⭐ Good | Real-time queries |
| **Hybrid** | `--hybrid` flag | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | Keyword + semantic |
| **Reranked** | `--rerank` flag | ⚡⚡⚡ Slower | ⭐⭐⭐⭐⭐ Best | Production ranking |

### Query Examples

**Strong Results:**
- "What is the treatment for diabetes?"
- "How to prevent cholera transmission?"
- "Causes of lung cancer"

**Decent Results:**
- "symptoms of malaria"
- "HIV infection methods"
- "COVID-19 prevention"

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

## 🚀 Advanced Usage

```bash
# Evaluate system performance
python evaluation.py --run-benchmark

# Check index statistics
python demo.py

# Custom search with options
python search_enhanced.py \
  --query "disease symptoms" \
  --hybrid \
  --rerank \
  --top-k 10
```

## 📝 Configuration

Edit `config/config.py` to customize:
- Embedding model and dimensions
- Chunk size (tokens)
- FAISS index type
- Search parameters
- Request delays and timeouts

## ⚙️ Requirements

See `requirements_embeddings.txt` for full dependencies. Main packages:
- `torch` - Deep learning framework
- `transformers` - HuggingFace models
- `faiss-cpu` - Vector indexing
- `rank-bm25` - BM25 ranking
- `beautifulsoup4` - Web scraping
- `requests` - HTTP library

## 📚 How It Works

1. **Scraper** fetches WHO fact sheets and extracts structured content
2. **Preprocessor** cleans text and splits into optimal chunks
3. **Embedder** converts text chunks to 768-dim vectors using Bio_ClinicalBERT
4. **Indexer** stores vectors in FAISS for fast retrieval
5. **Search Engine** finds relevant chunks and reranks results

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Large-scale web scraping and data engineering
- ✅ NLP preprocessing and text normalization  
- ✅ Semantic embeddings and vector databases
- ✅ Similarity search and ranking algorithms
- ✅ Production-ready system design
- ✅ Performance evaluation and metrics

## 📋 Course Information

**Course**: Big Data Analytics  
**Assignment**: Implement Vector Database with Semantic Search  
**Semester**: Fall 2025  
**Status**: ✅ Complete

## 📞 Notes

- First run will download the Bio_ClinicalBERT model (~1.5GB)
- FAISS can also use GPU acceleration (see `faiss-gpu` installation)
- Logs are saved to `logs/` directory for debugging
- All data sources and URLs are properly attributed

---

**Last Updated**: November 2025  
**Python**: 3.8+

