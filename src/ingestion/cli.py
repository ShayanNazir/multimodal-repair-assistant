import argparse
from pathlib import Path

from src.ingestion.pdf_processor import run_ingestion


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents for Multimodal Repair Assistant"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/documents", 
        help="Path to directory containing PDF files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/extracted", 
        help="Path to directory for extracted artifacts"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Starting ingestion from {input_path} to {output_path}...")
    processed = run_ingestion(input_path, output_path)
    
    if processed:
        print(f"Successfully processed {len(processed)} files: {', '.join(processed)}")
    else:
        print("No files were processed.")

if __name__ == "__main__":
    main()
