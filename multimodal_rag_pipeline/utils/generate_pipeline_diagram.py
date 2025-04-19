#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Diagram Generator

This script generates a visual diagram of the multimodal RAG pipeline architecture
using Graphviz.
"""

import os
import argparse
from pathlib import Path

def generate_pipeline_diagram(output_path=None, format='png'):
    """
    Generate a visual diagram of the multimodal RAG pipeline.
    
    Args:
        output_path: Path to save the diagram (default: pipeline_diagram.{format} in current directory)
        format: Output format (png, pdf, svg)
        
    Returns:
        Path to the generated diagram
    """
    try:
        import graphviz
    except ImportError:
        print("Graphviz Python package not installed. Please install with: pip install graphviz")
        print("Note: You also need to install the Graphviz software: https://graphviz.org/download/")
        return None
    
    # Create a new directed graph
    dot = graphviz.Digraph(
        'Multimodal_RAG_Pipeline', 
        comment='Multimodal RAG Pipeline Architecture',
        format=format,
        engine='dot'
    )
    
    # Set graph attributes
    dot.attr(rankdir='LR', size='12,8', ratio='fill', fontname='Arial', ranksep='1.5')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12', margin='0.2,0.1')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Define node colors
    colors = {
        'input': '#f5f5f5',
        'doc_proc': '#f9d5e5',
        'content_proc': '#d5f9e5',
        'embedding': '#d5e5f9',
        'retrieval': '#f9f9d5',
        'generation': '#e5d5f9',
        'output': '#f5f5f5'
    }
    
    # Create clusters for each pipeline stage
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input', style='filled', color=colors['input'], fontcolor='#333333')
        c.node('pdf', 'PDF Documents')
        c.node('docx', 'DOCX Documents')
        c.node('html', 'HTML Documents')
        c.node('images', 'Image Files')
        c.node('audio', 'Audio Files')
    
    with dot.subgraph(name='cluster_doc_proc') as c:
        c.attr(label='Document Processing', style='filled', color=colors['doc_proc'], fontcolor='#333333')
        c.node('doc_loader', 'Document Loaders')
        c.node('meta_extract', 'Metadata Extraction')
        c.node('layout', 'Layout Analysis')
        c.node('modality', 'Modality Extraction')
    
    with dot.subgraph(name='cluster_content_proc') as c:
        c.attr(label='Content Processing', style='filled', color=colors['content_proc'], fontcolor='#333333')
        c.node('text_proc', 'Text Processing')
        c.node('image_proc', 'Image Analysis')
        c.node('style_analysis', 'Artistic Style Analysis')
        c.node('fusion', 'Multimodal Fusion')
    
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Embedding & Indexing', style='filled', color=colors['embedding'], fontcolor='#333333')
        c.node('text_embed', 'Text Embeddings')
        c.node('image_embed', 'Image Embeddings')
        c.node('style_embed', 'Style Embeddings')
        c.node('vector_db', 'Vector Database')
    
    with dot.subgraph(name='cluster_retrieval') as c:
        c.attr(label='Retrieval & Generation', style='filled', color=colors['retrieval'], fontcolor='#333333')
        c.node('query', 'Query Understanding')
        c.node('retrieval', 'Hybrid Retrieval')
        c.node('llm', 'LLM Integration')
        c.node('response', 'Response Generation')
    
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='filled', color=colors['output'], fontcolor='#333333')
        c.node('final_response', 'Generated Response')
    
    # Add edges between nodes
    # Input to Document Processing
    dot.edge('pdf', 'doc_loader')
    dot.edge('docx', 'doc_loader')
    dot.edge('html', 'doc_loader')
    dot.edge('images', 'doc_loader')
    dot.edge('audio', 'doc_loader')
    
    # Document Processing flow
    dot.edge('doc_loader', 'meta_extract')
    dot.edge('meta_extract', 'layout')
    dot.edge('layout', 'modality')
    
    # Modality Extraction to Content Processing
    dot.edge('modality', 'text_proc')
    dot.edge('modality', 'image_proc')
    dot.edge('image_proc', 'style_analysis')
    
    # Content Processing flow
    dot.edge('text_proc', 'fusion')
    dot.edge('image_proc', 'fusion')
    dot.edge('style_analysis', 'fusion')
    
    # Content Processing to Embedding
    dot.edge('fusion', 'text_embed')
    dot.edge('fusion', 'image_embed')
    dot.edge('style_analysis', 'style_embed')
    
    # Embedding to Vector Database
    dot.edge('text_embed', 'vector_db')
    dot.edge('image_embed', 'vector_db')
    dot.edge('style_embed', 'vector_db')
    
    # Retrieval flow
    dot.edge('vector_db', 'retrieval')
    dot.edge('query', 'retrieval')
    dot.edge('retrieval', 'llm')
    dot.edge('llm', 'response')
    
    # Output
    dot.edge('response', 'final_response')
    
    # Add a special node for user query
    dot.node('user_query', 'User Query', shape='ellipse', style='filled', fillcolor='#f0f0f0')
    dot.edge('user_query', 'query')
    
    # Determine output path
    if output_path is None:
        output_path = f"pipeline_diagram.{format}"
    
    # Render the graph
    output_file = dot.render(filename=output_path, cleanup=True)
    print(f"Pipeline diagram generated: {output_file}")
    
    return output_file


def generate_style_analysis_diagram(output_path=None, format='png'):
    """
    Generate a visual diagram of the artistic style analysis component.
    
    Args:
        output_path: Path to save the diagram (default: style_analysis_diagram.{format} in current directory)
        format: Output format (png, pdf, svg)
        
    Returns:
        Path to the generated diagram
    """
    try:
        import graphviz
    except ImportError:
        print("Graphviz Python package not installed. Please install with: pip install graphviz")
        print("Note: You also need to install the Graphviz software: https://graphviz.org/download/")
        return None
    
    # Create a new directed graph
    dot = graphviz.Digraph(
        'Style_Analysis', 
        comment='Artistic Style Analysis Component',
        format=format,
        engine='dot'
    )
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='8,10', ratio='fill', fontname='Arial')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12', margin='0.2,0.1')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Define node colors
    colors = {
        'input': '#f5f5f5',
        'model': '#d5e5f9',
        'processing': '#f9d5e5',
        'output': '#d5f9e5',
        'integration': '#e5d5f9'
    }
    
    # Create nodes
    dot.node('artwork', 'Artwork Image', style='filled', fillcolor=colors['input'])
    dot.node('analyzer', 'Style Analyzer', style='filled', fillcolor=colors['processing'])
    
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Models', style='filled', color=colors['model'], fontcolor='#333333')
        c.node('clip', 'CLIP Model')
        c.node('vit', 'ViT Model')
    
    dot.node('embeddings', 'Style Embeddings', style='filled', fillcolor=colors['processing'])
    
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label='Analysis Outputs', style='filled', color=colors['output'], fontcolor='#333333')
        c.node('classification', 'Style Classification')
        c.node('features', 'Style Features')
        c.node('similarity', 'Style Similarity Matrix')
    
    with dot.subgraph(name='cluster_integration') as c:
        c.attr(label='RAG Integration', style='filled', color=colors['integration'], fontcolor='#333333')
        c.node('index', 'RAG Index')
        c.node('queries', 'Style-Based Queries')
        c.node('response', 'Enhanced Responses')
    
    # Add edges
    dot.edge('artwork', 'analyzer')
    dot.edge('analyzer', 'clip')
    dot.edge('analyzer', 'vit')
    dot.edge('clip', 'embeddings')
    dot.edge('vit', 'embeddings')
    dot.edge('embeddings', 'classification')
    dot.edge('embeddings', 'features')
    dot.edge('embeddings', 'similarity')
    dot.edge('classification', 'index')
    dot.edge('features', 'index')
    dot.edge('similarity', 'queries')
    dot.edge('index', 'queries')
    dot.edge('queries', 'response')
    
    # Determine output path
    if output_path is None:
        output_path = f"style_analysis_diagram.{format}"
    
    # Render the graph
    output_file = dot.render(filename=output_path, cleanup=True)
    print(f"Style analysis diagram generated: {output_file}")
    
    return output_file


def main():
    """Main function to parse arguments and generate diagrams."""
    parser = argparse.ArgumentParser(description='Generate pipeline diagrams')
    parser.add_argument('--output', '-o', help='Output directory for diagrams')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], default='png', help='Output format')
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate pipeline diagram
    pipeline_output = None
    if output_dir:
        pipeline_output = output_dir / f"pipeline_diagram"
    generate_pipeline_diagram(pipeline_output, args.format)
    
    # Generate style analysis diagram
    style_output = None
    if output_dir:
        style_output = output_dir / f"style_analysis_diagram"
    generate_style_analysis_diagram(style_output, args.format)


if __name__ == "__main__":
    main()