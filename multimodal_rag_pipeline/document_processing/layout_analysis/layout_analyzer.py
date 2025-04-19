#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layout Analyzer

This module provides functionality to analyze document layouts,
detecting structural elements like headers, footers, tables, and figures.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Analyzer for document layouts that detects structural elements.
    
    This class provides methods to:
    - Detect document sections (headers, footers, body)
    - Identify tables and their structures
    - Locate figures and captions
    - Map relationships between text blocks and visual positions
    
    It supports multiple layout analysis backends (LayoutParser, Detectron2, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the layout analyzer with configuration.
        
        Args:
            config: Configuration dictionary with settings for layout analysis
        """
        self.config = config
        self.engine = config.get("engine", "layoutparser")
        self.model_name = config.get("model", "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config")
        self.detect_tables = config.get("detect_tables", True)
        self.detect_figures = config.get("detect_figures", True)
        self.detect_headers = config.get("detect_headers", True)
        self.detect_footers = config.get("detect_footers", True)
        
        # Initialize the appropriate layout analysis engine
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize the layout analysis engine based on configuration."""
        if self.engine == "layoutparser":
            try:
                import layoutparser as lp
                
                # Load the model
                logger.info(f"Loading LayoutParser model: {self.model_name}")
                self.model = lp.Detectron2LayoutModel(
                    config_path=self.model_name,
                    label_map={
                        0: "Text",
                        1: "Title",
                        2: "List",
                        3: "Table",
                        4: "Figure"
                    }
                )
                logger.info("LayoutParser model loaded successfully")
                
            except ImportError:
                logger.error("LayoutParser not installed. Please install with: pip install layoutparser")
                raise
            except Exception as e:
                logger.error(f"Error initializing LayoutParser: {e}")
                raise
                
        elif self.engine == "unstructured":
            try:
                import unstructured
                # Unstructured initialization if needed
                logger.info("Using unstructured for layout analysis")
                
            except ImportError:
                logger.error("Unstructured not installed. Please install with: pip install unstructured")
                raise
                
        elif self.engine == "custom":
            # Custom layout analysis implementation
            logger.info("Using custom layout analysis implementation")
            
        else:
            logger.warning(f"Unknown layout analysis engine: {self.engine}. Falling back to basic analysis.")
    
    def analyze(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the layout of a document.
        
        Args:
            document: Document dictionary containing text, images, and other content
            
        Returns:
            Dictionary containing layout analysis results:
            {
                'sections': list of section dictionaries,
                'tables': list of table dictionaries,
                'figures': list of figure dictionaries,
                'text_blocks': list of text block dictionaries,
                'relationships': list of relationship dictionaries
            }
        """
        logger.info(f"Analyzing layout for document: {document.get('path', 'unknown')}")
        
        # Initialize result dictionary
        result = {
            'sections': [],
            'tables': [],
            'figures': [],
            'text_blocks': [],
            'relationships': []
        }
        
        # Analyze layout based on the configured engine
        if self.engine == "layoutparser":
            self._analyze_with_layoutparser(document, result)
        elif self.engine == "unstructured":
            self._analyze_with_unstructured(document, result)
        elif self.engine == "custom":
            self._analyze_with_custom(document, result)
        else:
            self._analyze_basic(document, result)
        
        logger.info(f"Layout analysis completed: {len(result['sections'])} sections, "
                   f"{len(result['tables'])} tables, {len(result['figures'])} figures, "
                   f"{len(result['text_blocks'])} text blocks")
        
        return result
    
    def _analyze_with_layoutparser(self, document: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Analyze document layout using LayoutParser.
        
        Args:
            document: Document dictionary
            result: Dictionary to populate with analysis results
        """
        import layoutparser as lp
        
        # Process each page
        for page_idx, page in enumerate(document.get('pages', [])):
            page_num = page_idx + 1
            
            # Check if we have an image for this page
            page_image = None
            for img in document.get('images', []):
                if img.get('page_num') == page_num and img.get('is_page_image', False):
                    # Convert image bytes to PIL Image
                    page_image = Image.open(io.BytesIO(img['image_data']))
                    break
            
            # If no page image is found, skip layout analysis for this page
            if page_image is None:
                logger.warning(f"No image found for page {page_num}, skipping layout analysis")
                continue
            
            # Convert PIL Image to numpy array if needed
            if not isinstance(page_image, np.ndarray):
                page_image = np.array(page_image)
            
            # Perform layout analysis
            layout = self.model.detect(page_image)
            
            # Process layout results
            for block_idx, block in enumerate(layout):
                block_id = f"page{page_num}_block{block_idx+1}"
                block_type = block.type
                block_coords = {
                    'x0': float(block.block.x_1),
                    'y0': float(block.block.y_1),
                    'x1': float(block.block.x_2),
                    'y1': float(block.block.y_2),
                }
                
                # Create text block entry
                text_block = {
                    'id': block_id,
                    'page_num': page_num,
                    'type': block_type,
                    'position': block_coords,
                    'confidence': float(block.score)
                }
                
                result['text_blocks'].append(text_block)
                
                # Process specific block types
                if block_type == "Title" and self.detect_headers:
                    # Add to sections as header
                    section = {
                        'id': f"header_{block_id}",
                        'page_num': page_num,
                        'type': 'header',
                        'position': block_coords,
                        'text_block_id': block_id
                    }
                    result['sections'].append(section)
                    
                elif block_type == "Table" and self.detect_tables:
                    # Add to tables
                    table = {
                        'id': f"table_{block_id}",
                        'page_num': page_num,
                        'position': block_coords,
                        'text_block_id': block_id
                    }
                    result['tables'].append(table)
                    
                elif block_type == "Figure" and self.detect_figures:
                    # Add to figures
                    figure = {
                        'id': f"figure_{block_id}",
                        'page_num': page_num,
                        'position': block_coords,
                        'text_block_id': block_id
                    }
                    result['figures'].append(figure)
            
            # Detect headers and footers based on position
            if self.detect_headers:
                # Simple heuristic: blocks in the top 15% of the page might be headers
                for block in layout:
                    if block.block.y_1 < page_image.shape[0] * 0.15:
                        block_id = f"page{page_num}_block{layout.index(block)+1}"
                        # Check if this block is already classified as a header
                        if not any(section['text_block_id'] == block_id for section in result['sections']):
                            section = {
                                'id': f"header_{block_id}",
                                'page_num': page_num,
                                'type': 'header',
                                'position': {
                                    'x0': float(block.block.x_1),
                                    'y0': float(block.block.y_1),
                                    'x1': float(block.block.x_2),
                                    'y1': float(block.block.y_2),
                                },
                                'text_block_id': block_id
                            }
                            result['sections'].append(section)
            
            if self.detect_footers:
                # Simple heuristic: blocks in the bottom 15% of the page might be footers
                for block in layout:
                    if block.block.y_2 > page_image.shape[0] * 0.85:
                        block_id = f"page{page_num}_block{layout.index(block)+1}"
                        # Check if this block is already classified
                        if not any(section['text_block_id'] == block_id for section in result['sections']):
                            section = {
                                'id': f"footer_{block_id}",
                                'page_num': page_num,
                                'type': 'footer',
                                'position': {
                                    'x0': float(block.block.x_1),
                                    'y0': float(block.block.y_1),
                                    'x1': float(block.block.x_2),
                                    'y1': float(block.block.y_2),
                                },
                                'text_block_id': block_id
                            }
                            result['sections'].append(section)
    
    def _analyze_with_unstructured(self, document: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Analyze document layout using Unstructured.
        
        Args:
            document: Document dictionary
            result: Dictionary to populate with analysis results
        """
        try:
            from unstructured.partition.pdf import partition_pdf
            
            # Get the document path
            doc_path = document.get('path')
            if not doc_path or not os.path.exists(doc_path):
                logger.error(f"Document path not found: {doc_path}")
                return
            
            # Process the PDF with Unstructured
            elements = partition_pdf(doc_path)
            
            # Process elements
            for elem_idx, element in enumerate(elements):
                elem_id = f"elem{elem_idx+1}"
                
                # Determine element type and create appropriate entries
                if hasattr(element, 'category'):
                    category = element.category
                    
                    # Create text block entry
                    text_block = {
                        'id': elem_id,
                        'type': category,
                        'text': str(element)
                    }
                    
                    # Add page number if available
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                        text_block['page_num'] = element.metadata.page_number
                    
                    result['text_blocks'].append(text_block)
                    
                    # Process specific element types
                    if category == "Title" or category == "Header":
                        section = {
                            'id': f"header_{elem_id}",
                            'type': 'header',
                            'text_block_id': elem_id
                        }
                        if 'page_num' in text_block:
                            section['page_num'] = text_block['page_num']
                        result['sections'].append(section)
                        
                    elif category == "Table":
                        table = {
                            'id': f"table_{elem_id}",
                            'text_block_id': elem_id
                        }
                        if 'page_num' in text_block:
                            table['page_num'] = text_block['page_num']
                        result['tables'].append(table)
                        
                    elif category == "Image" or category == "Figure":
                        figure = {
                            'id': f"figure_{elem_id}",
                            'text_block_id': elem_id
                        }
                        if 'page_num' in text_block:
                            figure['page_num'] = text_block['page_num']
                        result['figures'].append(figure)
                        
                    elif category == "Footer":
                        section = {
                            'id': f"footer_{elem_id}",
                            'type': 'footer',
                            'text_block_id': elem_id
                        }
                        if 'page_num' in text_block:
                            section['page_num'] = text_block['page_num']
                        result['sections'].append(section)
            
        except ImportError:
            logger.error("Unstructured not installed. Please install with: pip install unstructured")
        except Exception as e:
            logger.error(f"Error analyzing with Unstructured: {e}")
    
    def _analyze_with_custom(self, document: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Analyze document layout using custom implementation.
        
        Args:
            document: Document dictionary
            result: Dictionary to populate with analysis results
        """
        # Custom layout analysis implementation
        # This is a placeholder for custom implementation
        logger.info("Custom layout analysis not implemented yet")
    
    def _analyze_basic(self, document: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Perform basic layout analysis using heuristics.
        
        Args:
            document: Document dictionary
            result: Dictionary to populate with analysis results
        """
        # Process each page
        for page_idx, page in enumerate(document.get('pages', [])):
            page_num = page_idx + 1
            page_text = page.get('text', '')
            
            # Split text into lines
            lines = page_text.split('\n')
            
            # Basic heuristic: first line might be a header
            if lines and self.detect_headers:
                header_text = lines[0]
                header = {
                    'id': f"page{page_num}_header",
                    'page_num': page_num,
                    'type': 'header',
                    'text': header_text
                }
                result['sections'].append(header)
                
                # Create text block for header
                text_block = {
                    'id': f"page{page_num}_block1",
                    'page_num': page_num,
                    'type': 'Title',
                    'text': header_text
                }
                result['text_blocks'].append(text_block)
            
            # Basic heuristic: last line might be a footer
            if lines and len(lines) > 1 and self.detect_footers:
                footer_text = lines[-1]
                footer = {
                    'id': f"page{page_num}_footer",
                    'page_num': page_num,
                    'type': 'footer',
                    'text': footer_text
                }
                result['sections'].append(footer)
                
                # Create text block for footer
                text_block = {
                    'id': f"page{page_num}_block2",
                    'page_num': page_num,
                    'type': 'Text',
                    'text': footer_text
                }
                result['text_blocks'].append(text_block)
            
            # Basic table detection: look for lines with multiple delimiters
            if self.detect_tables:
                table_lines = []
                in_table = False
                table_start_idx = 0
                
                for line_idx, line in enumerate(lines):
                    # Heuristic: lines with multiple pipe or tab characters might be tables
                    if '|' in line and line.count('|') >= 3:
                        if not in_table:
                            in_table = True
                            table_start_idx = line_idx
                        table_lines.append(line)
                    elif '\t' in line and line.count('\t') >= 2:
                        if not in_table:
                            in_table = True
                            table_start_idx = line_idx
                        table_lines.append(line)
                    else:
                        if in_table and len(table_lines) >= 2:
                            # We've found a table
                            table = {
                                'id': f"page{page_num}_table{len(result['tables'])+1}",
                                'page_num': page_num,
                                'text': '\n'.join(table_lines),
                                'start_line': table_start_idx,
                                'end_line': line_idx - 1
                            }
                            result['tables'].append(table)
                            
                            # Create text block for table
                            text_block = {
                                'id': f"page{page_num}_block{len(result['text_blocks'])+1}",
                                'page_num': page_num,
                                'type': 'Table',
                                'text': '\n'.join(table_lines)
                            }
                            result['text_blocks'].append(text_block)
                            
                        in_table = False
                        table_lines = []
                
                # Check if we ended while still in a table
                if in_table and len(table_lines) >= 2:
                    table = {
                        'id': f"page{page_num}_table{len(result['tables'])+1}",
                        'page_num': page_num,
                        'text': '\n'.join(table_lines),
                        'start_line': table_start_idx,
                        'end_line': len(lines) - 1
                    }
                    result['tables'].append(table)
                    
                    # Create text block for table
                    text_block = {
                        'id': f"page{page_num}_block{len(result['text_blocks'])+1}",
                        'page_num': page_num,
                        'type': 'Table',
                        'text': '\n'.join(table_lines)
                    }
                    result['text_blocks'].append(text_block)


# Factory function to create a layout analyzer
def create_layout_analyzer(config: Dict[str, Any]) -> LayoutAnalyzer:
    """
    Create a layout analyzer with the specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LayoutAnalyzer instance
    """
    return LayoutAnalyzer(config)