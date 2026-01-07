#!/usr/bin/env python3
"""
Demonstration Script for WHO Vector Database

Automated demonstration of all system capabilities:
- Basic vector search
- Hybrid search (Vector + BM25)
- Cross-encoder reranking
- Evaluation metrics

Usage:
    python demo.py
    python demo.py --full  # Include all enhancements
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from config.config import LOGS_DIR, LOG_FORMAT, LOG_DATE_FORMAT

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Configure logging."""
    log_file = LOGS_DIR / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_basic_search():
    """Demonstrate basic vector search."""
    print("\n" + "=" * 80)
    print("DEMO 1: BASIC VECTOR SEARCH")
    print("=" * 80)
    
    from search_index import load_search_components, perform_search, format_results
    
    # Load components
    print("\nLoading search components...")
    index, metadata, metadata_list, model, tokenizer = load_search_components()
    
    # Demo queries
    queries = [
        "What are the symptoms of COVID-19?",
        "How is tuberculosis transmitted?",
        "What is the treatment for diabetes?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'-' * 80}")
        print(f"Query {i}: {query}")
        print("-" * 80)
        
        results = perform_search(query, model, tokenizer, index, metadata, metadata_list, top_k=3)
        
        for r in results:
            print(f"\n{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
            print(f"   {r['text'][:150]}...")
    
    print("\n" + "=" * 80)
    print("✓ Basic search demo complete")
    print("=" * 80)


def demo_hybrid_search():
    """Demonstrate hybrid search."""
    print("\n" + "=" * 80)
    print("DEMO 2: HYBRID SEARCH (Vector + BM25)")
    print("=" * 80)
    
    from search_index import load_search_components
    from search_enhanced import enhanced_search
    from utils.hybrid_search import BM25Index
    
    # Load components
    print("\nLoading search components...")
    index, metadata, metadata_list, model, tokenizer = load_search_components()
    
    print("Building BM25 index...")
    bm25_index = BM25Index(metadata_list)
    
    # Demo query
    query = "tuberculosis transmission"
    
    print(f"\nQuery: {query}")
    print("\n" + "-" * 80)
    print("Vector Search Only:")
    print("-" * 80)
    
    from search_index import perform_search
    vector_results = perform_search(query, model, tokenizer, index, metadata, metadata_list, top_k=3)
    
    for r in vector_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    print("\n" + "-" * 80)
    print("Hybrid Search (Vector + BM25):")
    print("-" * 80)
    
    hybrid_results = enhanced_search(
        query, model, tokenizer, index, metadata, metadata_list,
        bm25_index=bm25_index,
        use_hybrid=True,
        top_k=3
    )
    
    for r in hybrid_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
        if 'vector_score' in r and 'bm25_score' in r:
            print(f"   Vector: {r['vector_score']:.3f}, BM25: {r['bm25_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✓ Hybrid search demo complete")
    print("=" * 80)


def demo_reranking():
    """Demonstrate reranking."""
    print("\n" + "=" * 80)
    print("DEMO 3: CROSS-ENCODER RERANKING")
    print("=" * 80)
    
    from search_index import load_search_components, perform_search
    from utils.reranker import CrossEncoderReranker
    
    # Load components
    print("\nLoading search components...")
    index, metadata, metadata_list, model, tokenizer = load_search_components()
    
    print("Loading cross-encoder...")
    cross_encoder = CrossEncoderReranker()
    
    # Demo query
    query = "What are lung cancer symptoms?"
    
    print(f"\nQuery: {query}")
    print("\n" + "-" * 80)
    print("Before Reranking:")
    print("-" * 80)
    
    results = perform_search(query, model, tokenizer, index, metadata, metadata_list, top_k=5)
    
    for r in results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    print("\n" + "-" * 80)
    print("After Cross-Encoder Reranking:")
    print("-" * 80)
    
    reranked = cross_encoder.rerank(query, results, top_k=5)
    
    for r in reranked:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
        if 'original_score' in r:
            print(f"   Original: {r['original_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✓ Reranking demo complete")
    print("=" * 80)


def demo_evaluation():
    """Demonstrate evaluation metrics."""
    print("\n" + "=" * 80)
    print("DEMO 4: EVALUATION METRICS")
    print("=" * 80)
    
    from evaluation import run_benchmark_evaluation, print_evaluation_report
    from search_index import load_search_components
    
    # Load components
    print("\nLoading search components...")
    index, metadata, metadata_list, model, tokenizer = load_search_components()
    
    print("\nRunning benchmark evaluation (this may take a few minutes)...")
    results = run_benchmark_evaluation(model, tokenizer, index, metadata, metadata_list)
    
    print_evaluation_report(results)
    
    print("\n" + "=" * 80)
    print("✓ Evaluation demo complete")
    print("=" * 80)


def demo_comparison():
    """Compare all search methods side-by-side."""
    print("\n" + "=" * 80)
    print("DEMO 5: METHOD COMPARISON")
    print("=" * 80)
    
    from search_index import load_search_components, perform_search
    from search_enhanced import enhanced_search
    from utils.hybrid_search import BM25Index
    from utils.reranker import CrossEncoderReranker
    
    # Load all components
    print("\nLoading all components...")
    index, metadata, metadata_list, model, tokenizer = load_search_components()
    
    print("Building BM25 index...")
    bm25_index = BM25Index(metadata_list)
    
    print("Loading cross-encoder...")
    cross_encoder = CrossEncoderReranker()
    
    # Test query
    query = "malaria prevention methods"
    
    print(f"\nQuery: {query}")
    
    # Method 1: Basic vector search
    print("\n" + "-" * 80)
    print("Method 1: Basic Vector Search")
    print("-" * 80)
    
    basic_results = perform_search(query, model, tokenizer, index, metadata, metadata_list, top_k=3)
    for r in basic_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    # Method 2: Hybrid search
    print("\n" + "-" * 80)
    print("Method 2: Hybrid Search (Vector + BM25)")
    print("-" * 80)
    
    hybrid_results = enhanced_search(
        query, model, tokenizer, index, metadata, metadata_list,
        bm25_index=bm25_index,
        use_hybrid=True,
        top_k=3
    )
    for r in hybrid_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    # Method 3: With reranking
    print("\n" + "-" * 80)
    print("Method 3: Vector Search + Cross-Encoder Reranking")
    print("-" * 80)
    
    rerank_results = enhanced_search(
        query, model, tokenizer, index, metadata, metadata_list,
        cross_encoder=cross_encoder,
        use_reranking=True,
        top_k=3
    )
    for r in rerank_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    # Method 4: Full enhancement
    print("\n" + "-" * 80)
    print("Method 4: Full Enhancement (Hybrid + Reranking)")
    print("-" * 80)
    
    full_results = enhanced_search(
        query, model, tokenizer, index, metadata, metadata_list,
        bm25_index=bm25_index,
        cross_encoder=cross_encoder,
        use_hybrid=True,
        use_reranking=True,
        top_k=3
    )
    for r in full_results:
        print(f"{r['rank']}. {r['title']} (Score: {r['score']:.3f})")
    
    print("\n" + "=" * 80)
    print("✓ Comparison demo complete")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demonstrate vector database capabilities")
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all demos including evaluation (slower)"
    )
    
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Run only basic search demo"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("\n" + "=" * 80)
    print("WHO VECTOR DATABASE - SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases all system capabilities.")
    print("Please wait while components are loaded...")
    
    try:
        if args.basic_only:
            # Run only basic demo
            demo_basic_search()
        
        elif args.full:
            # Run all demos
            demo_basic_search()
            demo_hybrid_search()
            demo_reranking()
            demo_comparison()
            demo_evaluation()
        
        else:
            # Run main demos (skip evaluation for speed)
            demo_basic_search()
            demo_hybrid_search()
            demo_reranking()
            demo_comparison()
            
            print("\n💡 Tip: Run with --full to include evaluation metrics")
        
        print("\n" + "=" * 80)
        print("✓ ALL DEMOS COMPLETE")
        print("=" * 80)
        print("\nSystem Features Demonstrated:")
        print("  ✓ Vector similarity search (Bio_ClinicalBERT)")
        print("  ✓ Hybrid search (Vector + BM25 keyword matching)")
        print("  ✓ Cross-encoder reranking")
        if args.full:
            print("  ✓ Evaluation metrics (Precision@K, Recall@K, MAP)")
        print("\nFor interactive use, run:")
        print("  python search_index.py              # Basic search")
        print("  python search_enhanced.py --hybrid --rerank  # Enhanced search")
        print("\n" + "=" * 80)
        
        logger.info("✓ Demo completed successfully")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


