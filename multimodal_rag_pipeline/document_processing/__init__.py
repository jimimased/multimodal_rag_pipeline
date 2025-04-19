"""
Document Processing Components

This package contains components for document processing, including:
- Document loaders for various formats (PDF, DOCX, HTML)
- Metadata extraction
- Layout analysis
- Modality extraction (text, images, audio)
"""

from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def load_documents(input_dir, config):
    """
    Load documents from a directory.
    
    Args:
        input_dir: Directory containing documents
        config: Configuration dictionary
        
    Returns:
        List of document dictionaries
    """
    from multimodal_rag_pipeline.document_processing.document_loaders.pdf_loader import create_pdf_loader
    
    logger.info(f"Loading documents from {input_dir}")
    
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    documents = []
    
    # Process each file in the directory
    for file_path in input_dir.glob("**/*"):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            # Process based on file extension
            if file_ext in ['.pdf']:
                # Load PDF document
                pdf_loader = create_pdf_loader(config.get("document_loaders", {}).get("pdf", {}))
                document = pdf_loader.load(file_path)
                documents.append(document)
                
            elif file_ext in ['.docx', '.doc']:
                # TODO: Implement DOCX loader
                logger.warning(f"DOCX loader not implemented yet: {file_path}")
                
            elif file_ext in ['.html', '.htm']:
                # TODO: Implement HTML loader
                logger.warning(f"HTML loader not implemented yet: {file_path}")
                
            else:
                logger.info(f"Unsupported file type: {file_path}")
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents