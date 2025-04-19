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
import glob

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


def load_image_from_path(image_path):
    """Load an image from a file path."""
    try:
        image = Image.open(image_path)
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return image, image_data
    except Exception as e:
        print(f"Failed to load image from {image_path}: {e}")
        return None, None


def load_images_from_gdrive(gdrive_path, output_dir=None):
    """
    Load images from a Google Drive directory.
    
    Args:
        gdrive_path: Path to the Google Drive directory
        output_dir: Optional directory to save copies of the images
        
    Returns:
        List of image dictionaries
    """
    # Check if we're running in Google Colab
    try:
        from google.colab import drive
        is_colab = True
    except ImportError:
        is_colab = False
    
    if is_colab:
        # Mount Google Drive if not already mounted
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
    
    # Validate the path exists
    if not os.path.exists(gdrive_path):
        print(f"Google Drive path not found: {gdrive_path}")
        return []
    
    print(f"Loading images from Google Drive: {gdrive_path}")
    
    # Find all image files in the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(gdrive_path, ext)))
        image_files.extend(glob.glob(os.path.join(gdrive_path, '**', ext), recursive=True))
    
    # Sort image files for consistent ordering
    image_files.sort()
    
    # Load each image
    images = []
    for idx, image_path in enumerate(image_files):
        print(f"  Loading image {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Extract metadata from filename or path
        filename = os.path.basename(image_path)
        name_parts = os.path.splitext(filename)[0].split('_')
        
        # Try to extract artist and title from filename (if formatted as Artist_Title)
        if len(name_parts) >= 2:
            artist = name_parts[0].replace('-', ' ')
            title = ' '.join(name_parts[1:]).replace('-', ' ')
        else:
            artist = "Unknown"
            title = name_parts[0].replace('-', ' ')
        
        # Determine style from directory name if possible
        parent_dir = os.path.basename(os.path.dirname(image_path))
        style = parent_dir if parent_dir != os.path.basename(gdrive_path) else "Unknown"
        
        # Load the image
        image, image_data = load_image_from_path(image_path)
        
        if image is not None:
            # Save a copy to the output directory if specified
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{idx+1:02d}_{filename}")
                image.save(output_path)
            
            # Create image dictionary
            image_dict = {
                'id': f"artwork_{idx+1}",
                'title': title,
                'artist': artist,
                'style': style,
                'image_path': str(output_path) if output_path else str(image_path),
                'original_path': str(image_path),
                'image_data': image_data
            }
            
            images.append(image_dict)
    
    print(f"Loaded {len(images)} images from Google Drive")
    return images


def main():
    """Main function to demonstrate artistic style analysis."""
    parser = argparse.ArgumentParser(description='Artistic Style Analysis Example')
    parser.add_argument('--output', '-o', default='output', help='Output directory for results')
    parser.add_argument('--gdrive', '-g', help='Google Drive directory path containing images (e.g., /content/drive/MyDrive/SUMBA)')
    parser.add_argument('--use-web', '-w', action='store_true', help='Download sample images from the web instead of using Google Drive')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for style analysis if available')
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
        "use_gpu": args.gpu  # Use GPU if specified
    }
    
    # Create the style analyzer
    print("\nInitializing style analyzer...")
    style_analyzer = create_style_analyzer(style_config)
    print(f"Style analyzer created with model: {style_config['model']}")
    print(f"Using GPU: {style_config['use_gpu']}")
    
    # Load artwork images
    artwork_images = []
    
    # Option 1: Load from Google Drive
    if args.gdrive and not args.use_web:
        artwork_images = load_images_from_gdrive(args.gdrive, output_dir)
    
    # Option 2: Download from web
    elif args.use_web or not args.gdrive:
        print("\nDownloading sample artwork images from the web...")
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
    
    # Check if we have any images to analyze
    if not artwork_images:
        print("\nNo images found to analyze. Please provide a valid Google Drive path or use the --use-web option.")
        return
    
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