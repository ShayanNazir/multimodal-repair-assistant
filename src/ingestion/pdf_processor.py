import json
import logging
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PageMetadata(BaseModel):
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Extracted text from the page")
    char_count: int = Field(..., description="Number of characters in extracted text")
    image_path: Path = Field(..., description="Path to the rendered page image")

class DocumentMetadata(BaseModel):
    document_id: str
    source_filename: str
    total_pages: int
    pages: list[PageMetadata]

def process_pdf(pdf_path: Path, output_root: Path) -> DocumentMetadata | None:
    """
    Processes a single PDF: extracts text and renders pages as PNGs.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF {pdf_path}: {e}")
        return None

    doc_name = pdf_path.stem
    doc_output_dir = output_root / doc_name
    pages_dir = doc_output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages_meta = []
    total_pages = doc.page_count
    
    for page_index in range(total_pages):
        page_num = page_index + 1
        page = doc.load_page(page_index)
        
        # Extract text
        text = page.get_text().strip()
        
        # Render page to image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        image_filename = f"page_{page_num:04d}.png"
        image_path = pages_dir / image_filename
        pix.save(str(image_path))
        
        pages_meta.append(PageMetadata(
            page_number=page_num,
            text=text,
            char_count=len(text),
            image_path=image_path
        ))
    
    doc.close()
    
    return DocumentMetadata(
        document_id=doc_name,
        source_filename=pdf_path.name,
        total_pages=total_pages,
        pages=pages_meta
    )

def run_ingestion(input_dir: Path, output_dir: Path) -> list[str]:
    """
    Processes all PDFs in the input directory.
    """
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist.")
        return []

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {input_dir}.")
        return []

    processed_docs = []
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")
        meta = process_pdf(pdf_path, output_dir)
        if meta:
            meta_file = output_dir / f"{meta.document_id}_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json_data = meta.model_dump()
                def convert_paths(obj):
                    if isinstance(obj, dict):
                        return {k: convert_paths(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_paths(i) for i in obj]
                    elif isinstance(obj, Path):
                        return str(obj)
                    return obj
                
                f.write(json.dumps(convert_paths(json_data), indent=2))
            
            processed_docs.append(pdf_path.name)
            logger.info(f"Successfully processed {pdf_path.name}")

    return processed_docs
