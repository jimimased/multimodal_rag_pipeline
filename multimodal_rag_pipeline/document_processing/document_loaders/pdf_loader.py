#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Document Loader

This module provides functionality to load and process PDF documents,
extracting text, images, and structural information.
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loader for PDF documents that extracts text, images, and structural information.
    
    This class provides methods to:
    - Extract text while preserving layout
    - Extract images with position information
    - Detect and extract tables
    - Analyze document structure (headers, footers, etc.)
    - Extract metadata
    
    It supports multiple PDF processing backends (PyMuPDF, pdfplumber, etc.)
    and can fall back to OCR for scanned documents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PDF loader with configuration.
        
        Args:
            config: Configuration dictionary with settings for PDF processing
        """
        self.config = config
        self.engine = config.get("engine", "pypdf")
        self.extract_images = config.get("extract_images", True)
        self.ocr_fallback = config.get("ocr_fallback", True)
        
        # Initialize OCR if needed
        if self.ocr_fallback:
            try:
                import pytesseract
                self.pytesseract = pytesseract
            except ImportError:
                logger.warning("pytesseract not installed. OCR fallback will not be available.")
                self.ocr_fallback = False
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a PDF document and extract its content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing the extracted content:
            {
                'type': 'pdf',
                'path': str,
                'metadata': dict,
                'text': str,
                'pages': list of page dictionaries,
                'images': list of image dictionaries,
                'tables': list of table dictionaries,
                'structure': dict with structural information
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Loading PDF document: {file_path}")
        
        # Initialize result dictionary
        result = {
            'type': 'pdf',
            'path': str(file_path),
            'metadata': {},
            'text': '',
            'pages': [],
            'images': [],
            'tables': [],
            'structure': {}
        }
        
        # Extract content based on the configured engine
        if self.engine == "pymupdf":
            self._extract_with_pymupdf(file_path, result)
        elif self.engine == "pdfplumber":
            self._extract_with_pdfplumber(file_path, result)
        else:  # Default to PyPDF
            self._extract_with_pypdf(file_path, result)
        
        # Check if text extraction was successful, fall back to OCR if needed
        if not result['text'] and self.ocr_fallback:
            logger.info(f"No text extracted from {file_path}. Falling back to OCR.")
            self._extract_with_ocr(file_path, result)
        
        logger.info(f"Successfully loaded PDF document: {file_path}")
        return result
    
    def _extract_with_pymupdf(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Extract PDF content using PyMuPDF (fitz).
        
        Args:
            file_path: Path to the PDF file
            result: Dictionary to populate with extracted content
        """
        try:
            doc = fitz.open(file_path)
            
            # Extract metadata
            result['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': len(doc),
            }
            
            # Process each page
            full_text = []
            for page_idx, page in enumerate(doc):
                page_dict = {'page_num': page_idx + 1, 'text': '', 'images': [], 'tables': []}
                
                # Extract text
                page_text = page.get_text("text")
                page_dict['text'] = page_text
                full_text.append(page_text)
                
                # Extract images if configured
                if self.extract_images:
                    image_list = page.get_images(full=True)
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Get image position
                        for img_rect in page.get_image_rects(xref):
                            pos = {
                                'x0': img_rect.x0,
                                'y0': img_rect.y0,
                                'x1': img_rect.x1,
                                'y1': img_rect.y1,
                                'width': img_rect.width,
                                'height': img_rect.height,
                            }
                            
                            # Create image entry
                            image_entry = {
                                'id': f"page{page_idx+1}_img{img_idx+1}",
                                'page_num': page_idx + 1,
                                'position': pos,
                                'format': image_ext,
                                'image_data': image_bytes
                            }
                            
                            page_dict['images'].append(image_entry)
                            result['images'].append(image_entry)
                
                # Add page to result
                result['pages'].append(page_dict)
            
            # Combine all text
            result['text'] = "\n\n".join(full_text)
            
            # Close the document
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting content with PyMuPDF: {e}")
            raise
    
    def _extract_with_pdfplumber(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Extract PDF content using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            result: Dictionary to populate with extracted content
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                result['metadata'] = {
                    'page_count': len(pdf.pages),
                    'metadata': pdf.metadata
                }
                
                # Process each page
                full_text = []
                for page_idx, page in enumerate(pdf.pages):
                    page_dict = {'page_num': page_idx + 1, 'text': '', 'images': [], 'tables': []}
                    
                    # Extract text
                    page_text = page.extract_text()
                    page_dict['text'] = page_text
                    full_text.append(page_text)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        table_dict = {
                            'id': f"page{page_idx+1}_table{table_idx+1}",
                            'page_num': page_idx + 1,
                            'data': table
                        }
                        page_dict['tables'].append(table_dict)
                        result['tables'].append(table_dict)
                    
                    # Extract images (pdfplumber doesn't directly support image extraction)
                    # We'll use PyMuPDF for this if needed
                    
                    # Add page to result
                    result['pages'].append(page_dict)
                
                # Combine all text
                result['text'] = "\n\n".join(full_text)
                
        except Exception as e:
            logger.error(f"Error extracting content with pdfplumber: {e}")
            raise
    
    def _extract_with_pypdf(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Extract PDF content using PyPDF.
        
        Args:
            file_path: Path to the PDF file
            result: Dictionary to populate with extracted content
        """
        try:
            import pypdf
            
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                
                # Extract metadata
                result['metadata'] = {
                    'title': pdf.metadata.get('/Title', ''),
                    'author': pdf.metadata.get('/Author', ''),
                    'subject': pdf.metadata.get('/Subject', ''),
                    'creator': pdf.metadata.get('/Creator', ''),
                    'producer': pdf.metadata.get('/Producer', ''),
                    'page_count': len(pdf.pages),
                }
                
                # Process each page
                full_text = []
                for page_idx, page in enumerate(pdf.pages):
                    page_dict = {'page_num': page_idx + 1, 'text': '', 'images': [], 'tables': []}
                    
                    # Extract text
                    page_text = page.extract_text()
                    page_dict['text'] = page_text
                    full_text.append(page_text)
                    
                    # Add page to result
                    result['pages'].append(page_dict)
                
                # Combine all text
                result['text'] = "\n\n".join(full_text)
                
        except Exception as e:
            logger.error(f"Error extracting content with PyPDF: {e}")
            raise
    
    def _extract_with_ocr(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Extract PDF content using OCR.
        
        Args:
            file_path: Path to the PDF file
            result: Dictionary to populate with extracted content
        """
        if not self.ocr_fallback:
            logger.warning("OCR fallback requested but OCR is not available.")
            return
        
        try:
            import pdf2image
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(file_path)
            
            # Process each page
            full_text = []
            for page_idx, image in enumerate(images):
                page_dict = {'page_num': page_idx + 1, 'text': '', 'images': [], 'tables': []}
                
                # Perform OCR
                page_text = self.pytesseract.image_to_string(image)
                page_dict['text'] = page_text
                full_text.append(page_text)
                
                # Save the page image
                if self.extract_images:
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Create image entry
                    image_entry = {
                        'id': f"page{page_idx+1}_img1",
                        'page_num': page_idx + 1,
                        'position': {
                            'x0': 0,
                            'y0': 0,
                            'x1': image.width,
                            'y1': image.height,
                            'width': image.width,
                            'height': image.height,
                        },
                        'format': 'png',
                        'image_data': img_byte_arr
                    }
                    
                    page_dict['images'].append(image_entry)
                    result['images'].append(image_entry)
                
                # Add page to result
                result['pages'].append(page_dict)
            
            # Combine all text
            result['text'] = "\n\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Error extracting content with OCR: {e}")
            raise


# Factory function to create a PDF loader
def create_pdf_loader(config: Dict[str, Any]) -> PDFLoader:
    """
    Create a PDF loader with the specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PDFLoader instance
    """
    return PDFLoader(config)