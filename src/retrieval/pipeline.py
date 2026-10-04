import logging
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingGenerator
from .qdrant_store import QdrantStore
from .text_chunker import extract_chunks_from_metadata

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Main pipeline for indexing and retrieving text chunks."""
    
    def __init__(self, 
                 extracted_data_path: str,
                 storage_path: str,
                 embedding_generator: EmbeddingGenerator,
                 collection_name: str = "text_chunks"):
        """
        Initialize the retrieval pipeline.
        
        Args:
            extracted_data_path: Path to Phase 1 extracted data
            storage_path: Path for Qdrant storage
            embedding_generator: Embedding generator to use
            collection_name: Name of the Qdrant collection
        """
        self.extracted_data_path = Path(extracted_data_path)
        self.storage_path = Path(storage_path)
        self.embedding_generator = embedding_generator
        self.collection_name = collection_name
        
        # Initialize Qdrant store
        self.qdrant_store = QdrantStore(
            storage_path=str(self.storage_path),
            collection_name=collection_name
        )
    
    def index_documents(self):
        """
        Index all extracted documents:
        1. Read metadata files
        2. Chunk text
        3. Generate embeddings
        4. Store in Qdrant
        """
        logger.info(f"Starting document indexing from {self.extracted_data_path}")
        
        # Find all metadata files
        metadata_files = list(self.extracted_data_path.glob("*_metadata.json"))
        if not metadata_files:
            logger.warning(f"No metadata files found in {self.extracted_data_path}")
            return
        
        logger.info(f"Found {len(metadata_files)} metadata files to process")
        
        # Process each metadata file
        all_chunks = []
        for metadata_file in metadata_files:
            logger.info(f"Processing {metadata_file.name}")
            chunks = extract_chunks_from_metadata(str(metadata_file))
            all_chunks.extend(chunks)
            logger.info(f"  Extracted {len(chunks)} chunks")
        
        if not all_chunks:
            logger.warning("No chunks extracted from documents")
            return
        
        logger.info(f"Total chunks to index: {len(all_chunks)}")
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in all_chunks]
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Prepare payloads for Qdrant
        payloads = []
        for chunk in all_chunks:
            payload = {
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'source_file': chunk.source_file,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'text': chunk.text,
                'character_count': chunk.character_count,
                'image_path': chunk.image_path
            }
            payloads.append(payload)
        
        # Create/recreate Qdrant collection
        logger.info("Creating Qdrant collection...")
        self.qdrant_store.create_collection(dimension=self.embedding_generator.dimension)
        
        # Index all points
        logger.info("Indexing chunks in Qdrant...")
        self.qdrant_store.upsert_points(
            embeddings=embeddings,
            payloads=payloads
        )
        
        logger.info(f"Successfully indexed {self.qdrant_store.count()} chunks")
    
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: Natural language query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            return []
        
        # Generate embedding for query
        query_embeddings = self.embedding_generator.generate_embeddings([query])
        if not query_embeddings:
            return []
        
        query_vector = query_embeddings[0]
        
        # Search Qdrant
        results = self.qdrant_store.search(query_vector=query_vector, limit=top_k)
        
        # Format results to include similarity score and relevant fields
        formatted_results = []
        for result in results:
            payload = result['payload']
            formatted_result = {
                'score': result['score'],
                'chunk_id': payload['chunk_id'],
                'document_id': payload['document_id'],
                'source_file': payload['source_file'],
                'page_number': payload['page_number'],
                'chunk_index': payload['chunk_index'],
                'text': payload['text'],
                'character_count': payload['character_count'],
                'image_path': payload['image_path']
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
