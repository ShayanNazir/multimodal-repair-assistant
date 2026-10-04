import argparse
import json
import logging
import sys
from pathlib import Path

from .embeddings import DeterministicFakeEmbeddingGenerator, SentenceTransformerEmbeddingGenerator
from .pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_pipeline(use_fake_embeddings: bool = False) -> RetrievalPipeline:
    """
    Create a retrieval pipeline instance.
    
    Args:
        use_fake_embeddings: Whether to use fake embeddings (for testing)
        
    Returns:
        Configured RetrievalPipeline instance
    """
    # Paths
    extracted_data_path = "data/extracted"
    storage_path = "storage/qdrant"
    
    # Create embedding generator
    if use_fake_embeddings:
        embedding_generator = DeterministicFakeEmbeddingGenerator(dimension=384)
        logger.info("Using deterministic fake embedding generator")
    else:
        embedding_generator = SentenceTransformerEmbeddingGenerator()
        logger.info("Using SentenceTransformer embedding generator")
    
    # Create pipeline
    pipeline = RetrievalPipeline(
        extracted_data_path=extracted_data_path,
        storage_path=storage_path,
        embedding_generator=embedding_generator
    )
    
    return pipeline


def cmd_index(args):
    """Handle the index command."""
    setup_logging()
    logger.info("Starting indexing process...")
    
    # Determine if we should use fake embeddings
    use_fake = getattr(args, 'fake', False)
    
    pipeline = create_pipeline(use_fake_embeddings=use_fake)
    pipeline.index_documents()
    
    logger.info("Indexing completed successfully!")


def cmd_search(args):
    """Handle the search command."""
    setup_logging()
    
    pipeline = create_pipeline(use_fake_embeddings=args.fake)
    
    logger.info(f"Searching for: '{args.query}'")
    results = pipeline.search(query=args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results for: '{args.query}'\n")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Document: {result['source_file']}")
        print(f"   Page: {result['page_number']}")
        print(f"   Text: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        print()


def cmd_evaluate(args):
    """Handle the evaluation command."""
    setup_logging()
    
    # Load benchmark data
    benchmark_path = Path("data/evaluation/benchmark.json")
    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found: {benchmark_path}")
        logger.info("Please create a benchmark file at data/evaluation/benchmark.json")
        return
    
    with open(benchmark_path) as f:
        benchmark_data = json.load(f)
    
    pipeline = create_pipeline(use_fake_embeddings=args.fake)
    
    # Calculate metrics
    recall_at_k = {}
    mrr_sum = 0.0
    total_queries = len(benchmark_data)
    
    logger.info(f"Evaluating {total_queries} queries...")
    
    for item in benchmark_data:
        query = item['query']
        expected_pages = set(item['expected_pages'])
        
        # Search for results
        results = pipeline.search(query=query, top_k=args.top_k)
        
        # Calculate Recall@K for different K values
        for k in [1, 3, 5]:
            if k not in recall_at_k:
                recall_at_k[k] = 0.0
            
            # Check if any expected page is in top k results
            top_k_pages = {r['page_number'] for r in results[:k]}
            if expected_pages & top_k_pages:  # Intersection is not empty
                recall_at_k[k] += 1.0
        
        # Calculate MRR
        # Find the rank of the first relevant result
        for rank, result in enumerate(results, 1):
            if result['page_number'] in expected_pages:
                mrr_sum += 1.0 / rank
                break
        # If no relevant result found, MRR contribution is 0 (already accounted for)
    
    # Calculate averages
    for k in recall_at_k:
        recall_at_k[k] /= total_queries
    
    mrr = mrr_sum / total_queries if total_queries > 0 else 0.0
    
    # Print results
    print("\nRetrieval Evaluation Results")
    print("=" * 40)
    print(f"Number of queries: {total_queries}")
    print(f"Top-K considered: {args.top_k}")
    print()
    
    print("Recall@K:")
    for k in sorted(recall_at_k.keys()):
        print(f"  Recall@{k}: {recall_at_k[k]:.3f}")
    
    print(f"MRR: {mrr:.3f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal Repair Assistant - Retrieval CLI"
    )
    parser.add_argument(
        '--fake',
        action='store_true',
        help='Use fake deterministic embeddings (useful for testing)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index extracted documents')
    index_parser.set_defaults(func=cmd_index)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for text')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    search_parser.set_defaults(func=cmd_search)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run retrieval evaluation')
    eval_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to consider for evaluation (default: 5)'
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
