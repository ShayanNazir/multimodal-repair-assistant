from pathlib import Path
from typing import Any


class QdrantStore:
    """Interface for Qdrant vector storage and retrieval."""
    
    def __init__(self, storage_path: str, collection_name: str = "text_chunks"):
        """
        Initialize Qdrant store in local mode.
        
        Args:
            storage_path: Path where Qdrant will store data
            collection_name: Name of the collection to use
        """
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.client = None
        self._initialized = False
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _ensure_initialized(self):
        """Lazy initialization of Qdrant client."""
        if self._initialized:
            return
            
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
            
            self.client = QdrantClient(path=str(self.storage_path))
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct
            self._initialized = True
            
        except ImportError as err:
            raise ImportError(
                "qdrant-client package is required for QdrantStore. "
                "Install it with: pip install qdrant-client"
            ) from err
    
    def create_collection(self, dimension: int):
        """
        Create or recreate a collection for storing vectors.
        
        Args:
            dimension: Dimensionality of the vectors
        """
        self._ensure_initialized()
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            # Collection might not exist, which is fine
            pass
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.VectorParams(
                size=dimension,
                distance=self.Distance.COSINE
            )
        )
    
    def upsert_points(self, 
                      embeddings: list[list[float]], 
                      payloads: list[dict[str, Any]],
                      ids: list[int] | None = None):
        """
        Insert or update points in the collection.
        
        Args:
            embeddings: List of embedding vectors
            payloads: List of metadata payloads for each vector
            ids: Optional list of integer IDs (if None, auto-generated)
        """
        self._ensure_initialized()
        
        if not embeddings:
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = list(range(len(embeddings)))
        
        # Create points
        points = []
        for i, (embedding, payload) in enumerate(zip(embeddings, payloads, strict=False)):
            point = self.PointStruct(
                id=ids[i],
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, 
               query_vector: list[float], 
               limit: int = 5) -> list[dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and payloads
        """
        self._ensure_initialized()
        
        if not query_vector:
            return []
        
        # Check if collection exists, if not return empty results
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            # Collection doesn't exist yet
            return []
        
        # Perform search using query_points method
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                'score': result.score,
                'payload': result.payload
            })
        
        return formatted_results
    
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        self._ensure_initialized()
        try:
            return self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            # Collection doesn't exist
            return {"status": "not_found"}
    
    def count(self) -> int:
        """Get the number of points in the collection."""
        self._ensure_initialized()
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            # Collection doesn't exist
            return 0
