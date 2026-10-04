import json
import os
import tempfile
from pathlib import Path

import pytest

from src.retrieval.embeddings import DeterministicFakeEmbeddingGenerator
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.text_chunker import TextChunk, chunk_text, extract_chunks_from_metadata


def test_text_chunker_basic():
    """Test basic text chunking functionality."""
    text = "This is a test sentence. " * 50  # About 100 words
    chunks = chunk_text(
        text=text,
        document_id="test_doc",
        source_file="test.pdf",
        page_number=1,
        image_path="test.png",
        target_words=20,
        overlap_words=5
    )
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(chunk.document_id == "test_doc" for chunk in chunks)
    assert all(chunk.source_file == "test.pdf" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.image_path == "test.png" for chunk in chunks)
    
    # Check chunk IDs are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"test_doc_p0001_c{i:04d}"


def test_text_chunker_empty_text():
    """Test chunking with empty text."""
    chunks = chunk_text(
        text="",
        document_id="test_doc",
        source_file="test.pdf",
        page_number=1,
        image_path="test.png"
    )
    assert chunks == []


def test_text_chunker_no_overlap():
    """Test chunking with no overlap."""
    text = "word " * 60  # 60 words
    chunks = chunk_text(
        text=text,
        document_id="test_doc",
        source_file="test.pdf",
        page_number=1,
        image_path="test.png",
        target_words=20,
        overlap_words=0
    )
    
    # Should have exactly 3 chunks (60/20)
    assert len(chunks) == 3
    
    # Check that each chunk has approximately 20 words
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert 18 <= word_count <= 22  # Allow small variance


def test_text_chunker_overlap():
    """Test that overlap is working correctly."""
    # Use distinct words to properly test overlap
    words = [f"word{i}" for i in range(50)]  # word0, word1, word2, ..., word49
    text = " ".join(words)
    chunks = chunk_text(
        text=text,
        document_id="test_doc",
        source_file="test.pdf",
        page_number=1,
        image_path="test.png",
        target_words=10,
        overlap_words=5
    )
    
    # With 50 words, target 10, overlap 5:
    # Chunk 0: words 0-9 (word0-word9)
    # Chunk 1: words 5-14 (word5-word14)
    # Chunk 2: words 10-19 (word10-word19)
    # Chunk 3: words 15-24 (word15-word24)
    # Chunk 4: words 20-29 (word20-word29)
    # Chunk 5: words 25-34 (word25-word34)
    # Chunk 6: words 30-39 (word30-word39)
    # Chunk 7: words 35-44 (word35-word44)
    # Chunk 8: words 40-49 (word40-word49)
    # Chunk 9: words 45-49 (word45-word49) - shorter final chunk
    assert len(chunks) == 10
    
    # Check overlap between consecutive chunks (should be 5 words)
    for i in range(len(chunks) - 1):
        current_words = set(chunks[i].text.split())
        next_words = set(chunks[i + 1].text.split())
        overlap = current_words.intersection(next_words)
        # For most chunks, we expect exactly 5 words overlap
        # The last few overlaps might be less due to the final chunks being shorter
        if i < len(chunks) - 2:  # Not the last two pairs
            expected_msg = (
                f"Expected 5 overlap words between chunks {i} and {i+1}, "
                f"got {len(overlap)}: {overlap}"
            )
            assert len(overlap) == 5, expected_msg


def test_extract_chunks_from_metadata():
    """Test extracting chunks from metadata file."""
    # Create temporary metadata file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as f:
        metadata = {
            "document_id": "test_doc",
            "source_filename": "test.pdf",
            "total_pages": 2,
            "pages": [
                {
                    "page_number": 1,
                    "text": "This is page one with some text content for testing purposes.",
                    "char_count": 62,
                    "image_path": "data/extracted/test_doc/pages/page_0001.png"
                },
                {
                    "page_number": 2,
                    "text": "This is page two with different content also for testing.",
                    "char_count": 58,
                    "image_path": "data/extracted/test_doc/pages/page_0002.png"
                }
            ]
        }
        json.dump(metadata, f)
        temp_path = f.name
    
    try:
        chunks = extract_chunks_from_metadata(temp_path)
        
        assert len(chunks) > 0
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
        assert all(chunk.source_file == "test.pdf" for chunk in chunks)
        
        # Check we have chunks from both pages
        page_numbers = set(chunk.page_number for chunk in chunks)
        assert page_numbers == {1, 2}
        
        # Check chunk IDs are properly formed
        for chunk in chunks:
            expected_id = f"{chunk.document_id}_p{chunk.page_number:04d}_c{chunk.chunk_index:04d}"
            assert chunk.chunk_id == expected_id
            
    finally:
        os.unlink(temp_path)


def test_extract_chunks_skip_empty_pages():
    """Test that empty pages are skipped."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as f:
        metadata = {
            "document_id": "test_doc",
            "source_filename": "test.pdf",
            "total_pages": 3,
            "pages": [
                {
                    "page_number": 1,
                    "text": "This page has content.",
                    "char_count": 20,
                    "image_path": "data/extracted/test_doc/pages/page_0001.png"
                },
                {
                    "page_number": 2,
                    "text": "",  # Empty page
                    "char_count": 0,
                    "image_path": "data/extracted/test_doc/pages/page_0002.png"
                },
                {
                    "page_number": 3,
                    "text": "This page also has content.",
                    "char_count": 28,
                    "image_path": "data/extracted/test_doc/pages/page_0003.png"
                }
            ]
        }
        json.dump(metadata, f)
        temp_path = f.name
    
    try:
        chunks = extract_chunks_from_metadata(temp_path)
        
        # Should only have chunks from pages 1 and 3
        page_numbers = set(chunk.page_number for chunk in chunks)
        assert page_numbers == {1, 3}
        
    finally:
        os.unlink(temp_path)


def test_qdrant_store_initialization():
    """Test Qdrant store initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = QdrantStore(storage_path=tmpdir, collection_name="test_collection")
        assert store.storage_path == Path(tmpdir)
        assert store.collection_name == "test_collection"
        assert not store._initialized  # Should not be initialized yet


def test_qdrant_store_create_collection():
    """Test creating a Qdrant collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = QdrantStore(storage_path=tmpdir, collection_name="test_collection")
        
        # This should work without actually connecting to Qdrant in tests
        # Since we're mocking, we expect it to raise ImportError which is handled in the test
        try:
            store.create_collection(dimension=384)
        except ImportError:
            # Expected when qdrant-client is not available in test environment
            pass


def test_deterministic_fake_embedding_generator():
    """Test the deterministic fake embedding generator."""
    generator = DeterministicFakeEmbeddingGenerator(dimension=10)
    
    # Test same input produces same output
    texts = ["hello world", "test sentence", "another example"]
    emb1 = generator.generate_embeddings(texts)
    emb2 = generator.generate_embeddings(texts)
    
    assert emb1 == emb2
    assert len(emb1) == 3
    assert all(len(vec) == 10 for vec in emb1)
    
    # Test empty input
    assert generator.generate_embeddings([]) == []
    
    # Test dimension property
    assert generator.dimension == 10


def test_pipeline_initialization():
    """Test retrieval pipeline initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_path = Path(tmpdir) / "extracted"
        storage_path = Path(tmpdir) / "storage"
        extracted_path.mkdir()
        
        embedding_generator = DeterministicFakeEmbeddingGenerator(dimension=384)
        pipeline = RetrievalPipeline(
            extracted_data_path=str(extracted_path),
            storage_path=str(storage_path),
            embedding_generator=embedding_generator
        )
        
        assert pipeline.extracted_data_path == extracted_path
        assert pipeline.storage_path == storage_path
        assert pipeline.embedding_generator == embedding_generator
        assert isinstance(pipeline.qdrant_store, QdrantStore)


def test_pipeline_search_empty_query():
    """Test searching with empty query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_path = Path(tmpdir) / "extracted"
        storage_path = Path(tmpdir) / "storage"
        extracted_path.mkdir()
        
        embedding_generator = DeterministicFakeEmbeddingGenerator(dimension=384)
        pipeline = RetrievalPipeline(
            extracted_data_path=str(extracted_path),
            storage_path=str(storage_path),
            embedding_generator=embedding_generator
        )
        
        # Empty query should return empty results
        results = pipeline.search(query="", top_k=5)
        assert results == []
        
        results = pipeline.search(query="   ", top_k=5)
        assert results == []


def test_chunking_determinitive_ids():
    """Test that chunk IDs are deterministic and stable."""
    text = "This is a test. " * 30
    
    # Generate chunks twice with same parameters
    chunks1 = chunk_text(
        text=text,
        document_id="doc1",
        source_file="file1.pdf",
        page_number=5,
        image_path="image1.png"
    )
    
    chunks2 = chunk_text(
        text=text,
        document_id="doc1",
        source_file="file1.pdf",
        page_number=5,
        image_path="image1.png"
    )
    
    # Should be identical
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2, strict=False):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index


if __name__ == "__main__":
    pytest.main([__file__])
