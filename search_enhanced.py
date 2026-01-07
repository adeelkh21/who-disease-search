#!/usr/bin/env python3
"""
Enhanced FAISS Vector Search with Hybrid Search and Reranking

Advanced semantic search with:
- Hybrid search (Vector + BM25)
- Cross-encoder reranking
- Query expansion
- Multiple ranking strategies

Usage:
    # Basic search
    python search_enhanced.py --query "What are COVID-19 symptoms?"
    
    # Hybrid search
    python search_enhanced.py --query "tuberculosis transmission" --hybrid
    
    # With reranking
    python search_enhanced.py --query "diabetes treatment" --rerank
    
    # Full enhancement (hybrid + reranking)
    python search_enhanced.py --query "malaria symptoms" --hybrid --rerank
    
    # Interactive mode
    python search_enhanced.py --hybrid --rerank

Dependencies:
    pip install -r requirements_embeddings.txt
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from config.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
    DEFAULT_TOP_K,
    LOGS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)

from utils.embeddings import load_model_and_tokenizer
from utils.io_utils import load_embeddings_with_metadata
from utils.search_utils import load_faiss_index, load_index_metadata
from utils.hybrid_search import BM25Index, hybrid_search, expand_medical_query
from utils.reranker import CrossEncoderReranker, rerank_by_diversity
from search_index import perform_search, format_results


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_file = LOGS_DIR / f"search_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# ENHANCED SEARCH
# =============================================================================

def enhanced_search(query: str,
                   model,
                   tokenizer,
                   index,
                   metadata: Dict,
                   metadata_list: List[Dict],
                   bm25_index: BM25Index = None,
                   cross_encoder: CrossEncoderReranker = None,
                   use_hybrid: bool = False,
                   use_reranking: bool = False,
                   use_diversity: bool = False,
                   use_expansion: bool = False,
                   alpha: float = 0.7,
                   top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Perform enhanced search with optional hybrid search and reranking.
    
    Args:
        query: Query string
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata list
        bm25_index: BM25 index (for hybrid search)
        cross_encoder: Cross-encoder reranker
        use_hybrid: Enable hybrid search
        use_reranking: Enable cross-encoder reranking
        use_diversity: Enable diversity-based reranking
        use_expansion: Enable query expansion
        alpha: Weight for hybrid search (vector vs BM25)
        top_k: Number of results
        
    Returns:
        List of search results
    """
    logger = logging.getLogger(__name__)
    
    # Query expansion
    queries = [query]
    if use_expansion:
        queries = expand_medical_query(query)
        logger.info(f"Query expansion: {len(queries)} variations")
    
    # Perform vector search for all query variations
    all_results = []
    for q in queries:
        results = perform_search(q, model, tokenizer, index, metadata, metadata_list, top_k=top_k*2)
        all_results.extend(results)
    
    # Deduplicate and merge scores
    merged_results = {}
    for result in all_results:
        doc_id = result['index_id']
        if doc_id not in merged_results:
            merged_results[doc_id] = result
        else:
            # Keep higher score
            if result['score'] > merged_results[doc_id]['score']:
                merged_results[doc_id] = result
    
    results = list(merged_results.values())
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k*2]
    
    # Hybrid search
    if use_hybrid and bm25_index:
        logger.info("Applying hybrid search (Vector + BM25)")
        results = hybrid_search(query, results, bm25_index, alpha=alpha, top_k=top_k*2)
    
    # Diversity reranking
    if use_diversity:
        logger.info("Applying diversity-based reranking")
        results = rerank_by_diversity(results, diversity_weight=0.3)
    
    # Cross-encoder reranking
    if use_reranking and cross_encoder:
        logger.info("Applying cross-encoder reranking")
        results = cross_encoder.rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]
    
    # Update ranks
    for rank, result in enumerate(results, start=1):
        result['rank'] = rank
    
    return results


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(model, tokenizer, index, metadata, metadata_list,
                    bm25_index=None, cross_encoder=None,
                    use_hybrid=False, use_reranking=False):
    """
    Enhanced interactive search mode.
    
    Args:
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata
        bm25_index: BM25 index
        cross_encoder: Cross-encoder reranker
        use_hybrid: Enable hybrid search
        use_reranking: Enable reranking
    """
    print("\n" + "=" * 80)
    print("🔍 ENHANCED INTERACTIVE SEARCH MODE")
    print("=" * 80)
    print(f"Features: Hybrid={use_hybrid}, Reranking={use_reranking}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Perform enhanced search
            results = enhanced_search(
                query, model, tokenizer, index, metadata, metadata_list,
                bm25_index=bm25_index,
                cross_encoder=cross_encoder,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking
            )
            
            # Display results
            print(format_results(results, format_type="text", query=query))
            print()
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logging.error(f"Search error: {e}", exc_info=True)
            print(f"❌ Error: {e}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced semantic search with hybrid search and reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results (default: {DEFAULT_TOP_K})"
    )
    
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (Vector + BM25)"
    )
    
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    
    parser.add_argument(
        "--diversity",
        action="store_true",
        help="Enable diversity-based reranking"
    )
    
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Enable query expansion"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid search weight for vector (default: 0.7)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "json", "compact"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    try:
        # Print header
        print("\n" + "=" * 80)
        print("WHO DISEASE FACT SHEETS - ENHANCED SEMANTIC SEARCH")
        print("=" * 80)
        
        # Load components
        logger.info("Loading search components...")
        from search_index import load_search_components
        
        index, metadata, metadata_list, model, tokenizer = load_search_components()
        
        # Initialize BM25 index if hybrid search enabled
        bm25_index = None
        if args.hybrid:
            logger.info("Building BM25 index for hybrid search...")
            bm25_index = BM25Index(metadata_list)
        
        # Initialize cross-encoder if reranking enabled
        cross_encoder = None
        if args.rerank:
            logger.info("Loading cross-encoder for reranking...")
            cross_encoder = CrossEncoderReranker()
        
        # Print info
        print(f"\n✓ Index: {index.ntotal} vectors")
        print(f"✓ Model: {metadata.get('model_name', 'Unknown')}")
        print(f"✓ Hybrid search: {'Enabled' if args.hybrid else 'Disabled'}")
        print(f"✓ Reranking: {'Enabled' if args.rerank else 'Disabled'}")
        print(f"✓ Diversity: {'Enabled' if args.diversity else 'Disabled'}")
        print(f"✓ Expansion: {'Enabled' if args.expand else 'Disabled'}")
        
        # Run mode
        if args.query:
            # Single query mode
            results = enhanced_search(
                args.query, model, tokenizer, index, metadata, metadata_list,
                bm25_index=bm25_index,
                cross_encoder=cross_encoder,
                use_hybrid=args.hybrid,
                use_reranking=args.rerank,
                use_diversity=args.diversity,
                use_expansion=args.expand,
                alpha=args.alpha,
                top_k=args.top_k
            )
            
            # Output results
            output = format_results(results, format_type=args.format, query=args.query)
            print(output)
        
        else:
            # Interactive mode
            interactive_mode(
                model, tokenizer, index, metadata, metadata_list,
                bm25_index=bm25_index,
                cross_encoder=cross_encoder,
                use_hybrid=args.hybrid,
                use_reranking=args.rerank
            )
        
        logger.info("✓ Search completed successfully")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


