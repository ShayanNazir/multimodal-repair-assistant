from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int
    image_path: str


def chunk_text(
    text: str,
    document_id: str,
    source_file: str,
    page_number: int,
    image_path: str,
    target_words: int = 200,
    overlap_words: int = 35
) -> list[TextChunk]:
    """
    Split text into overlapping chunks with deterministic IDs.
    
    Args:
        text: The text to chunk
        document_id: ID of the source document
        source_file: Original source filename
        page_number: Page number (1-indexed)
        image_path: Path to the page image
        target_words: Target words per chunk (180-220)
        overlap_words: Overlap between chunks (30-40)
        
    Returns:
        List of TextChunk objects
    """
    if not text.strip():
        return []
    
    # Split text into words
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start_idx = 0
    chunk_index = 0
    
    while start_idx < len(words):
        # Calculate end index for this chunk
        end_idx = min(start_idx + target_words, len(words))
        
        # Extract chunk text
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        # Generate stable chunk ID
        chunk_id = f"{document_id}_p{page_number:04d}_c{chunk_index:04d}"
        
        # Create chunk object
        chunk = TextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            text=chunk_text,
            character_count=len(chunk_text),
            image_path=image_path
        )
        
        chunks.append(chunk)
        
        # Move start index for next chunk (accounting for overlap)
        start_idx += target_words - overlap_words
        chunk_index += 1
        
        # Break if we've processed all words
        if start_idx >= len(words):
            break
    
    return chunks


def extract_chunks_from_metadata(metadata_path: str) -> list[TextChunk]:
    """
    Extract all chunks from a Phase 1 metadata file.
    
    Args:
        metadata_path: Path to the metadata.json file
        
    Returns:
        List of all TextChunk objects from the document
    """
    import json
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    document_id = metadata['document_id']
    source_file = metadata['source_filename']
    
    all_chunks = []
    
    for page_data in metadata['pages']:
        page_number = page_data['page_number']
        text = page_data['text']
        image_path = page_data['image_path']
        
        # Skip empty pages
        if not text.strip():
            continue
            
        # Chunk the text from this page
        page_chunks = chunk_text(
            text=text,
            document_id=document_id,
            source_file=source_file,
            page_number=page_number,
            image_path=image_path
        )
        
        all_chunks.extend(page_chunks)
    
    return all_chunks
