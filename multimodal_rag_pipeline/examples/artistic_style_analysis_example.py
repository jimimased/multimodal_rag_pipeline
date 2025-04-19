#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Artistic Style Analysis Example

This script demonstrates how to use the multimodal RAG pipeline
to analyze artistic styles in images and incorporate style information
into the retrieval and generation process.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import io

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_rag_pipeline.content_processing.image_analysis.style_analyzer import (
    create_style_analyzer,
    create_style_similarity_matrix,
    find_similar_images
)
from multimodal_rag_pipeline.utils.config_loader import load_config


def download_image(url, filename=None):
    """Download an image from a URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        if filename:
            image.save(filename)
        return image, response.content
    else:
        print(f"Failed to download image: {url}")
        return None, None


def main():
    """Main function to demonstrate artistic style analysis."""
    parser = argparse.ArgumentParser(description='Artistic Style Analysis Example')
    parser.add_argument('--output', '-o', default='output', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Multimodal RAG Pipeline - Artistic Style Analysis Example")
    print("=" * 60)
    
    # Create a configuration for style analysis
    style_config = {
        "enabled": True,
        "model": "clip",  # Options: clip, vit, custom
        "use_gpu": False  # Set to True if you have a GPU
    }
    
    # Create the style analyzer
    print("\nInitializing style analyzer...")
    style_analyzer = create_style_analyzer(style_config)
    print(f"Style analyzer created with model: {style_config['model']}")
    
    # Sample artwork URLs with different styles
    print("\nDownloading sample artwork images...")
    artwork_data = [
        {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg", 
         "title": "Starry Night", "artist": "Vincent van Gogh", "style": "Post-Impressionism"},
        {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Monet_-_Impression%2C_Sunrise.jpg/1280px-Monet_-_Impression%2C_Sunrise.jpg", 
         "title": "Impression, Sunrise", "artist": "Claude Monet", "style": "Impressionism"},
        {"url": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/1280px-PicassoGuernica.jpg", 
         "title": "Guernica", "artist": "Pablo Picasso", "style": "Cubism"},
        {"url": "https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/The_Persistence_of_Memory.jpg/1280px-The_Persistence_of_Memory.jpg", 
         "title": "The Persistence of Memory", "artist": "Salvador Dalí", "style": "Surrealism"},
        {"url": "https://upload.wikimedia.org/wikipedia/en/thumb/9/95/Warhol-Campbell_Soup-1-screenprint-1968.jpg/1280px-Warhol-Campbell_Soup-1-screenprint-1968.jpg", 
         "title": "Campbell's Soup Cans", "artist": "Andy Warhol", "style": "Pop Art"}
    ]
    
    # Download and save the images
    artwork_images = []
    for idx, item in enumerate(artwork_data):
        filename = output_dir / f"{idx+1:02d}_{item['artist'].replace(' ', '_')}_{item['title'].replace(' ', '_')}.jpg"
        print(f"  Downloading {item['title']} by {item['artist']}...")
        
        image, image_data = download_image(item['url'], filename)
        
        if image is not None:
            # Create image dictionary
            image_dict = {
                'id': f"artwork_{idx+1}",
                'title': item['title'],
                'artist': item['artist'],
                'style': item['style'],
                'image_path': str(filename),
                'image_data': image_data
            }
            
            artwork_images.append(image_dict)
    
    print(f"\nDownloaded {len(artwork_images)} artwork images")
    
    # Analyze each artwork
    print("\nAnalyzing artistic styles...")
    for artwork in artwork_images:
        print(f"\n  Analyzing {artwork['title']} by {artwork['artist']} ({artwork['style']})...")
        
        # Analyze style
        style_analysis = style_analyzer.analyze(artwork)
        
        # Print style classification
        print("  Style classification:")
        for style, confidence in style_analysis['style_classification']:
            print(f"    {style}: {confidence:.2f}")
        
        # Store style analysis results in the artwork dictionary
        artwork['style_analysis'] = style_analysis
    
    # Create style similarity matrix
    print("\nCreating style similarity matrix...")
    similarity_data = create_style_similarity_matrix(artwork_images, style_analyzer)
    
    # Extract data for visualization
    similarity_matrix = np.array(similarity_data['similarity_matrix'])
    
    # Print similarity matrix
    print("\nStyle Similarity Matrix:")
    for i, artwork1 in enumerate(artwork_images):
        for j, artwork2 in enumerate(artwork_images):
            print(f"  {artwork1['artist']} - {artwork1['title']} vs {artwork2['artist']} - {artwork2['title']}: {similarity_matrix[i][j]:.2f}")
    
    # Find similar artworks for a query
    query_idx = 0  # Van Gogh's Starry Night
    query_artwork = artwork_images[query_idx]
    
    print(f"\nFinding artworks similar to {query_artwork['title']} by {query_artwork['artist']}...")
    similar_artworks = find_similar_images(query_artwork, artwork_images, style_analyzer)
    
    # Print similar artworks
    print("\nSimilar artworks:")
    for i, similar in enumerate(similar_artworks):
        print(f"  {i+1}. {similar['image']['title']} by {similar['image']['artist']} ({similar['image']['style']}) - Similarity: {similar['similarity']:.2f}")
    
    # Demonstrate RAG integration
    print("\nIntegrating with RAG Pipeline:")
    print("  1. Style embeddings can be indexed in the vector database")
    print("  2. Style-based queries can retrieve artworks with similar styles")
    print("  3. Style information can be incorporated into the generated responses")
    
    print("\nExample RAG query: 'Show me paintings similar to Van Gogh's style'")
    print("  1. Process query to identify style reference (Van Gogh)")
    print("  2. Retrieve artworks with similar style embeddings")
    print("  3. Generate response incorporating style information")
    
    # Example response
    print("\nExample response:")
    print("  Based on the style analysis, paintings similar to Van Gogh's style include:")
    for i, similar in enumerate(similar_artworks):
        print(f"  - {similar['image']['title']} by {similar['image']['artist']}, which shares {similar['similarity']:.0%} similarity in style")
    print("  Van Gogh's Post-Impressionism style is characterized by bold colors, expressive brushwork, and emotional intensity.")
    
    print("\nExample complete! Results saved to:", output_dir)


if __name__ == "__main__":
    main()