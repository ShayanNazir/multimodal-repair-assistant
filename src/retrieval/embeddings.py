from abc import ABC, abstractmethod


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generation."""
    
    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Real embedding generator using Sentence Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError as err:
            msg = (
                "sentence-transformers package is required for "
                "SentenceTransformerEmbeddingGenerator. "
                "Install it with: pip install sentence-transformers"
            )
            raise ImportError(msg) from err
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Sentence Transformers."""
        # Handle empty input
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        # Convert numpy arrays to lists
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class DeterministicFakeEmbeddingGenerator(EmbeddingGenerator):
    """Deterministic fake embedding generator for testing."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate deterministic fake embeddings based on text hash.
        Same input always produces same output.
        """
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            # Create deterministic vector based on text hash
            import hashlib
            
            # Use a simple hash-based approach for determinism
            hash_obj = hashlib.md5(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert bytes to list of floats in range [-1, 1]
            vector = []
            for i in range(min(len(hash_bytes), self._dimension)):
                # Convert byte to float in range [-1, 1]
                val = (hash_bytes[i] / 255.0) * 2 - 1
                vector.append(val)
            
            # Pad or truncate to desired dimension
            if len(vector) < self._dimension:
                vector.extend([0.0] * (self._dimension - len(vector)))
            else:
                vector = vector[:self._dimension]
                
            embeddings.append(vector)
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
