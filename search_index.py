#!/usr/bin/env python3
"""
FAISS Vector Search for WHO Disease Fact Sheets

Semantic search over medical text chunks using Bio_ClinicalBERT embeddings and FAISS.

Usage:
    # Interactive mode
    python search_index.py
    
    # Single query
    python search_index.py --query "What are the symptoms of COVID-19?" --top-k 5
    
    # Run test queries
    python search_index.py --run-tests
    
    # JSON output
    python search_index.py --query "diabetes treatment" --format json

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

import numpy as np

# Import configuration and utilities
from config.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
    DEFAULT_TOP_K,
    NORMALIZE_EMBEDDINGS,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOGS_DIR
)

from utils.embeddings import load_model_and_tokenizer, embed_text
from utils.io_utils import load_embeddings_with_metadata, validate_file_exists
from utils.search_utils import (
    load_faiss_index,
    load_index_metadata,
    search_index,
    l2_distance_to_cosine,
    format_results_text,
    format_results_compact
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: Path = None, verbose: bool = False):
    """Configure logging for the script."""
    log_file = log_file or LOGS_DIR / f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logs directory
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
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
# SEARCH OPERATIONS
# =============================================================================

def load_search_components():
    """
    Load all components needed for search.
    
    Returns:
        Tuple of (index, metadata, metadata_list, model, tokenizer)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Loading search components...")
    
    # Validate files exist
    validate_file_exists(FAISS_INDEX_PATH, "FAISS index")
    validate_file_exists(METADATA_PATH, "Index metadata")
    validate_file_exists(EMBEDDINGS_PATH, "Embeddings file")
    
    # Load FAISS index
    logger.info("Loading FAISS index...")
    index = load_faiss_index(FAISS_INDEX_PATH)
    
    # Load index metadata
    logger.info("Loading index metadata...")
    metadata = load_index_metadata(METADATA_PATH)
    
    # Load document metadata
    logger.info("Loading document metadata...")
    _, metadata_list = load_embeddings_with_metadata(EMBEDDINGS_PATH)
    
    # Load model
    logger.info("Loading Bio_ClinicalBERT model...")
    model, tokenizer = load_model_and_tokenizer()
    
    logger.info("✓ All components loaded successfully")
    
    return index, metadata, metadata_list, model, tokenizer


def perform_search(query: str,
                  model,
                  tokenizer,
                  index,
                  metadata: Dict,
                  metadata_list: List[Dict],
                  top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Perform semantic search for a query.
    
    Args:
        query: Query text
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata list
        top_k: Number of results to return
        
    Returns:
        List of result dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Check if embeddings are normalized
    normalized = metadata.get("normalized", NORMALIZE_EMBEDDINGS)
    
    # Generate query embedding
    logger.debug(f"Generating embedding for query: {query[:50]}...")
    query_embedding = embed_text(query, model, tokenizer, normalize=normalized)
    
    # Search index
    logger.debug(f"Searching for top {top_k} results...")
    distances, indices = search_index(query_embedding, index, top_k)
    
    # Build results
    results = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        if idx == -1:  # Invalid index
            continue
        
        if idx >= len(metadata_list):
            logger.warning(f"Index {idx} out of range, skipping")
            continue
        
        # Get document metadata
        doc = metadata_list[idx]
        
        # Convert distance to similarity score
        if normalized:
            score = l2_distance_to_cosine(dist)
        else:
            score = -dist  # Negative distance (higher is better)
        
        results.append({
            "rank": rank,
            "index_id": int(idx),
            "score": float(score),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "source_url": doc.get("source_url", "")
        })
    
    logger.info(f"Found {len(results)} results")
    return results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_results(results: List[Dict], 
                  format_type: str = "text",
                  query: str = None) -> str:
    """
    Format search results for output.
    
    Args:
        results: List of result dictionaries
        format_type: Output format ("text", "json", or "compact")
        query: Original query (for text format)
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)
    elif format_type == "compact":
        return format_results_compact(results)
    else:  # text
        return format_results_text(results, query=query)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(model, tokenizer, index, metadata, metadata_list):
    """
    Run interactive search prompt.
    
    Args:
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata list
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("🔍 INTERACTIVE SEARCH MODE")
    print("=" * 80)
    print("Enter your queries below. Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Perform search
            results = perform_search(
                query, model, tokenizer, index, metadata, 
                metadata_list, top_k=DEFAULT_TOP_K
            )
            
            # Display results
            print(format_results(results, format_type="text", query=query))
            print()
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            print(f"❌ Error: {e}\n")


# =============================================================================
# TEST QUERIES
# =============================================================================

def run_test_queries(model, tokenizer, index, metadata, metadata_list):
    """
    Run predefined test queries for validation.
    
    Args:
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata list
    """
    logger = logging.getLogger(__name__)
    
    test_queries = [
        "What are the symptoms of COVID-19?",
        "How is tuberculosis transmitted?",
        "What is the treatment for diabetes?"
    ]
    
    print("\n" + "=" * 80)
    print("🧪 RUNNING TEST QUERIES")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, start=1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_queries)}")
        
        # Perform search
        results = perform_search(
            query, model, tokenizer, index, metadata,
            metadata_list, top_k=3
        )
        
        # Display results
        print(format_results(results, format_type="text", query=query))
        
        # Sanity check
        if results:
            print(f"\n✅ Top result: \"{results[0]['title']}\" (score: {results[0]['score']:.3f})")
        else:
            print("\n⚠️  No results found")
    
    print("\n" + "=" * 80)
    print("✓ Test queries complete")
    print("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantic search over WHO disease fact sheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        help=f"Number of results to return (default: {DEFAULT_TOP_K})"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "json", "compact"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run predefined test queries"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--index",
        type=Path,
        default=FAISS_INDEX_PATH,
        help=f"Path to FAISS index (default: {FAISS_INDEX_PATH})"
    )
    
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help=f"Path to metadata file (default: {METADATA_PATH})"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    try:
        # Print header
        print("\n" + "=" * 80)
        print("WHO DISEASE FACT SHEETS - SEMANTIC SEARCH")
        print("=" * 80)
        
        # Load components
        index, metadata, metadata_list, model, tokenizer = load_search_components()
        
        # Print index info
        print(f"\n✓ Index loaded: {index.ntotal} vectors, dimension {index.d}")
        print(f"✓ Metadata loaded: {len(metadata_list)} entries")
        print(f"✓ Index type: {metadata.get('index_type', 'Unknown')}")
        print(f"✓ Normalized: {metadata.get('normalized', False)}")
        print(f"✓ Model: {metadata.get('model_name', 'Unknown')}")
        
        # Run appropriate mode
        if args.run_tests:
            # Test mode
            run_test_queries(model, tokenizer, index, metadata, metadata_list)
        
        elif args.query:
            # Single query mode
            results = perform_search(
                args.query, model, tokenizer, index, metadata,
                metadata_list, top_k=args.top_k
            )
            
            # Output results
            output = format_results(results, format_type=args.format, query=args.query)
            print(output)
        
        else:
            # Interactive mode
            interactive_mode(model, tokenizer, index, metadata, metadata_list)
        
        logger.info("✓ Search completed successfully")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
