#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal RAG Pipeline - Main Entry Point

This script serves as the main entry point for the multimodal RAG pipeline.
It orchestrates the document processing, content processing, embedding generation,
indexing, retrieval, and generation components.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_rag_pipeline.document_processing.document_loaders import load_documents
from multimodal_rag_pipeline.content_processing.text_processing import process_text
from multimodal_rag_pipeline.content_processing.image_analysis import analyze_images
from multimodal_rag_pipeline.content_processing.multimodal_fusion import fuse_modalities
from multimodal_rag_pipeline.embedding_indexing.text_embeddings import generate_text_embeddings
from multimodal_rag_pipeline.embedding_indexing.image_embeddings import generate_image_embeddings
from multimodal_rag_pipeline.embedding_indexing.vector_db import index_embeddings
from multimodal_rag_pipeline.retrieval_generation.query_understanding import process_query
from multimodal_rag_pipeline.retrieval_generation.llm_integration import generate_response
from multimodal_rag_pipeline.utils.config_loader import load_config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("multimodal_rag.log")
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing input documents"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to store output files"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["index", "query", "end_to_end"], 
        default="end_to_end",
        help="Pipeline execution mode"
    )
    parser.add_argument(
        "--query", 
        type=str,
        help="Query string (required in query mode)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the multimodal RAG pipeline."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Multimodal RAG Pipeline")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline based on mode
    if args.mode in ["index", "end_to_end"]:
        # Document processing
        logger.info("Starting document processing")
        documents = load_documents(args.input_dir, config["document_processing"])
        
        # Content processing
        logger.info("Starting content processing")
        processed_text = process_text(documents, config["content_processing"]["text"])
        processed_images = analyze_images(documents, config["content_processing"]["image"])
        fused_content = fuse_modalities(processed_text, processed_images, config["content_processing"]["fusion"])
        
        # Embedding generation and indexing
        logger.info("Generating embeddings and indexing")
        text_embeddings = generate_text_embeddings(processed_text, config["embedding"]["text"])
        image_embeddings = generate_image_embeddings(processed_images, config["embedding"]["image"])
        index_embeddings(text_embeddings, image_embeddings, fused_content, config["vector_db"])
        
        logger.info("Indexing completed successfully")
    
    if args.mode in ["query", "end_to_end"]:
        if not args.query and args.mode == "query":
            logger.error("Query string is required in query mode")
            sys.exit(1)
        
        query = args.query if args.query else input("Enter your query: ")
        
        # Query processing
        logger.info(f"Processing query: {query}")
        processed_query = process_query(query, config["retrieval"]["query"])
        
        # Response generation
        logger.info("Generating response")
        response = generate_response(processed_query, config["retrieval"]["llm"])
        
        print("\nResponse:")
        print(response)
    
    logger.info("Multimodal RAG Pipeline execution completed")


if __name__ == "__main__":
    main()