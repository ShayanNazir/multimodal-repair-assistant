import json

import pymupdf
import pytest

from src.ingestion.pdf_processor import process_pdf, run_ingestion


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a simple PDF file for testing."""
    pdf_path = tmp_path / "test_manual.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello Multimodal Assistant! This is a test page.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

def test_process_single_pdf(sample_pdf, tmp_path):
    output_dir = tmp_path / "extracted"
    meta = process_pdf(sample_pdf, output_dir)
    
    assert meta is not None
    assert meta.document_id == "test_manual"
    assert meta.total_pages == 1
    assert "Hello Multimodal Assistant!" in meta.pages[0].text
    assert meta.pages[0].image_path.exists()
    assert meta.pages[0].image_path.suffix == ".png"
    # Check for zero-padded filename
    assert meta.pages[0].image_path.name == "page_0001.png"

def test_run_ingestion_full_pipeline(sample_pdf, tmp_path):
    input_dir = tmp_path / "documents"
    input_dir.mkdir()
    # Move sample pdf to input dir
    test_pdf = input_dir / "test_manual.pdf"
    import shutil
    shutil.copy(sample_pdf, test_pdf)
    
    output_dir = tmp_path / "extracted"
    processed = run_ingestion(input_dir, output_dir)
    
    assert "test_manual.pdf" in processed
    
    # Check directory structure
    doc_dir = output_dir / "test_manual"
    assert doc_dir.exists()
    # Metadata file is saved in the extracted directory, not inside the doc directory
    assert (output_dir / "test_manual_metadata.json").exists()
    pages_dir = doc_dir / "pages"
    assert pages_dir.exists()
    assert (pages_dir / "page_0001.png").exists()
    
    # Check if JSON metadata was created (in the output_dir, not doc_dir)
    meta_file = output_dir / "test_manual_metadata.json"
    assert meta_file.exists()
    
    with open(meta_file) as f:
        data = json.load(f)
        assert data["document_id"] == "test_manual"
        assert data["total_pages"] == 1
        assert "Hello Multimodal Assistant!" in data["pages"][0]["text"]
        # Check that image path in metadata is correct
        assert data["pages"][0]["image_path"] == str(pages_dir / "page_0001.png")

def test_corrupt_pdf(tmp_path):
    # Create a file that is not a PDF
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_text("This is not a PDF")
    
    output_dir = tmp_path / "extracted"
    meta = process_pdf(corrupt_pdf, output_dir)
    
    assert meta is None

def test_empty_directory(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "extracted"
    
    processed = run_ingestion(input_dir, output_dir)
    assert processed == []

def test_missing_directory(tmp_path):
    input_dir = tmp_path / "non_existent"
    output_dir = tmp_path / "extracted"
    
    processed = run_ingestion(input_dir, output_dir)
    assert processed == []
