#!/usr/bin/env python3
"""
Evaluation Script for Vector Database Performance

Measures:
- Precision@K, Recall@K
- Mean Average Precision (MAP)
- Query response time
- Search accuracy

Usage:
    python evaluation.py --test-file test_queries.json
    python evaluation.py --run-benchmark
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

from config.config import (
    LOGS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    DEFAULT_TOP_K
)

from utils.embeddings import load_model_and_tokenizer
from utils.io_utils import load_embeddings_with_metadata
from utils.search_utils import load_faiss_index, load_index_metadata
from search_index import perform_search


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging."""
    log_file = LOGS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# EVALUATION METRICS
# =============================================================================

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# relevant items in top K) / K
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0-1)
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# relevant items in top K) / (total # relevant items)
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return relevant_retrieved / len(relevant)


def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate F1@K score.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        F1@K score (0-1)
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = (sum of Precision@K for each relevant item) / (# relevant items)
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: List of relevant document IDs (ground truth)
        
    Returns:
        Average Precision score (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    precision_sum = 0.0
    relevant_count = 0
    
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_sum += relevant_count / k
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant)


def mean_average_precision(results: List[Tuple[List[str], List[str]]]) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    MAP = average of AP scores across all queries
    
    Args:
        results: List of (retrieved, relevant) tuples for each query
        
    Returns:
        MAP score (0-1)
    """
    if len(results) == 0:
        return 0.0
    
    ap_scores = [average_precision(retrieved, relevant) for retrieved, relevant in results]
    
    return np.mean(ap_scores)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @K.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: List of relevant document IDs
        k: Number of top results
        
    Returns:
        NDCG@K score (0-1)
    """
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


# =============================================================================
# QUERY RESPONSE TIME ANALYSIS
# =============================================================================

def measure_query_time(query: str, model, tokenizer, index, metadata, metadata_list, 
                      top_k: int = 5) -> Tuple[List[Dict], float]:
    """
    Measure query execution time.
    
    Args:
        query: Query string
        model: Model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata
        top_k: Number of results
        
    Returns:
        Tuple of (results, execution_time_seconds)
    """
    start_time = time.time()
    results = perform_search(query, model, tokenizer, index, metadata, metadata_list, top_k)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return results, execution_time


# =============================================================================
# BENCHMARK EVALUATION
# =============================================================================

def run_benchmark_evaluation(model, tokenizer, index, metadata, metadata_list) -> Dict:
    """
    Run comprehensive benchmark evaluation.
    
    Args:
        model: Bio_ClinicalBERT model
        tokenizer: Tokenizer
        index: FAISS index
        metadata: Index metadata
        metadata_list: Document metadata list
        
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Predefined test queries with ground truth
    test_queries = [
        {
            "query": "What are the symptoms of COVID-19?",
            "relevant_terms": ["covid", "coronavirus", "sars-cov-2"],
            "expected_in_top_5": True
        },
        {
            "query": "How is tuberculosis transmitted?",
            "relevant_terms": ["tuberculosis", "tb", "mycobacterium"],
            "expected_in_top_5": True
        },
        {
            "query": "What is the treatment for diabetes?",
            "relevant_terms": ["diabetes", "insulin", "glucose"],
            "expected_in_top_5": True
        },
        {
            "query": "symptoms of malaria",
            "relevant_terms": ["malaria", "plasmodium", "fever"],
            "expected_in_top_5": True
        },
        {
            "query": "How to prevent cholera?",
            "relevant_terms": ["cholera", "prevention", "water"],
            "expected_in_top_5": True
        },
        {
            "query": "lung cancer causes",
            "relevant_terms": ["lung cancer", "tobacco", "smoking"],
            "expected_in_top_5": True
        },
        {
            "query": "HIV transmission methods",
            "relevant_terms": ["hiv", "aids", "transmission"],
            "expected_in_top_5": True
        },
        {
            "query": "dengue fever symptoms",
            "relevant_terms": ["dengue", "fever", "mosquito"],
            "expected_in_top_5": True
        },
        {
            "query": "hepatitis B prevention",
            "relevant_terms": ["hepatitis", "vaccine", "liver"],
            "expected_in_top_5": True
        },
        {
            "query": "asthma triggers",
            "relevant_terms": ["asthma", "respiratory", "breathing"],
            "expected_in_top_5": True
        }
    ]
    
    logger.info(f"Running benchmark evaluation with {len(test_queries)} queries...")
    
    results = {
        "total_queries": len(test_queries),
        "query_results": [],
        "precision_at_5": [],
        "recall_at_5": [],
        "f1_at_5": [],
        "response_times": [],
        "average_scores": []
    }
    
    for i, test in enumerate(test_queries, 1):
        logger.info(f"Query {i}/{len(test_queries)}: {test['query']}")
        
        # Perform search and measure time
        search_results, exec_time = measure_query_time(
            test['query'], model, tokenizer, index, metadata, metadata_list, top_k=5
        )
        
        # Extract retrieved document titles/IDs
        retrieved_titles = [r['title'].lower() for r in search_results]
        
        # Check if relevant terms appear in top results
        relevant_found = []
        for title in retrieved_titles:
            for term in test['relevant_terms']:
                if term.lower() in title:
                    relevant_found.append(title)
                    break
        
        # Calculate metrics (simplified - checking if relevant terms appear)
        precision = len(relevant_found) / 5 if search_results else 0.0
        recall = len(relevant_found) / len(test['relevant_terms']) if test['relevant_terms'] else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Average score of top 5
        avg_score = np.mean([r['score'] for r in search_results]) if search_results else 0.0
        
        results["precision_at_5"].append(precision)
        results["recall_at_5"].append(recall)
        results["f1_at_5"].append(f1)
        results["response_times"].append(exec_time)
        results["average_scores"].append(avg_score)
        
        query_result = {
            "query": test['query'],
            "top_result": search_results[0]['title'] if search_results else "N/A",
            "top_score": search_results[0]['score'] if search_results else 0.0,
            "precision@5": precision,
            "recall@5": recall,
            "f1@5": f1,
            "response_time_ms": exec_time * 1000,
            "avg_score": avg_score,
            "relevant_in_top_5": len(relevant_found) > 0
        }
        
        results["query_results"].append(query_result)
        
        logger.info(f"  Top result: {query_result['top_result']} (score: {query_result['top_score']:.3f})")
        logger.info(f"  Precision@5: {precision:.3f}, Recall@5: {recall:.3f}, F1@5: {f1:.3f}")
        logger.info(f"  Response time: {exec_time*1000:.2f} ms")
    
    # Calculate aggregate metrics
    results["mean_precision_at_5"] = np.mean(results["precision_at_5"])
    results["mean_recall_at_5"] = np.mean(results["recall_at_5"])
    results["mean_f1_at_5"] = np.mean(results["f1_at_5"])
    results["mean_response_time_ms"] = np.mean(results["response_times"]) * 1000
    results["median_response_time_ms"] = np.median(results["response_times"]) * 1000
    results["mean_score"] = np.mean(results["average_scores"])
    results["success_rate"] = sum(1 for r in results["query_results"] if r["relevant_in_top_5"]) / len(test_queries)
    
    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_evaluation_report(results: Dict):
    """
    Print formatted evaluation report.
    
    Args:
        results: Results dictionary from evaluation
    """
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\nTotal Queries Evaluated: {results['total_queries']}")
    print(f"Success Rate: {results['success_rate']*100:.1f}% (relevant result in top 5)")
    
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS")
    print("-" * 80)
    print(f"Mean Precision@5:  {results['mean_precision_at_5']:.4f}")
    print(f"Mean Recall@5:     {results['mean_recall_at_5']:.4f}")
    print(f"Mean F1@5:         {results['mean_f1_at_5']:.4f}")
    print(f"Mean Score:        {results['mean_score']:.4f}")
    
    print("\n" + "-" * 80)
    print("RESPONSE TIME ANALYSIS")
    print("-" * 80)
    print(f"Mean Response Time:   {results['mean_response_time_ms']:.2f} ms")
    print(f"Median Response Time: {results['median_response_time_ms']:.2f} ms")
    print(f"Min Response Time:    {min(results['response_times'])*1000:.2f} ms")
    print(f"Max Response Time:    {max(results['response_times'])*1000:.2f} ms")
    
    print("\n" + "-" * 80)
    print("PER-QUERY RESULTS")
    print("-" * 80)
    
    for i, qr in enumerate(results['query_results'], 1):
        print(f"\n{i}. Query: \"{qr['query']}\"")
        print(f"   Top Result: {qr['top_result']} (score: {qr['top_score']:.3f})")
        print(f"   Metrics: P@5={qr['precision@5']:.3f}, R@5={qr['recall@5']:.3f}, F1@5={qr['f1@5']:.3f}")
        print(f"   Time: {qr['response_time_ms']:.2f} ms | Relevant: {'✓' if qr['relevant_in_top_5'] else '✗'}")
    
    print("\n" + "=" * 80)


def save_evaluation_report(results: Dict, output_path: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate vector database performance"
    )
    
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run comprehensive benchmark evaluation"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_results.json"),
        help="Output path for results (default: data/evaluation_results.json)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("VECTOR DATABASE EVALUATION")
    logger.info("=" * 80)
    
    try:
        # Load components
        logger.info("Loading search components...")
        from search_index import load_search_components
        
        index, metadata, metadata_list, model, tokenizer = load_search_components()
        logger.info("✓ Components loaded")
        
        if args.run_benchmark:
            # Run benchmark
            results = run_benchmark_evaluation(model, tokenizer, index, metadata, metadata_list)
            
            # Print report
            print_evaluation_report(results)
            
            # Save results
            save_evaluation_report(results, args.output)
            
            logger.info("✓ Evaluation complete")
        else:
            print("Use --run-benchmark to run evaluation")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


