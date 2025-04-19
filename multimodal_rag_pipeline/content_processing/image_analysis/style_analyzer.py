#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Artistic Style Analyzer

This module provides functionality to analyze artistic styles in images
using multimodal embeddings and vision models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import io

logger = logging.getLogger(__name__)


class StyleAnalyzer:
    """
    Analyzer for artistic styles in images.
    
    This class provides methods to:
    - Generate embeddings for artistic style analysis
    - Classify images by artistic style
    - Compare images based on style similarity
    - Extract style-related features
    
    It supports multiple vision models (CLIP, ViT, etc.) and can be
    customized for specific artistic domains.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the style analyzer with configuration.
        
        Args:
            config: Configuration dictionary with settings for style analysis
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.model_name = config.get("model", "clip")
        self.device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
        
        # Initialize the appropriate style analysis model
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the style analysis model based on configuration."""
        if self.model_name == "clip":
            try:
                import clip
                
                logger.info(f"Loading CLIP model for style analysis on {self.device}")
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info("CLIP model loaded successfully")
                
                # Load style categories
                self.style_categories = [
                    "Impressionism", "Expressionism", "Cubism", "Surrealism", 
                    "Abstract", "Pop Art", "Minimalism", "Renaissance", "Baroque",
                    "Romanticism", "Realism", "Art Nouveau", "Art Deco", "Gothic",
                    "Neoclassicism", "Rococo", "Modernism", "Postmodernism", 
                    "Contemporary", "Digital Art", "Photography", "Illustration"
                ]
                
                # Encode style categories
                with torch.no_grad():
                    self.style_text_features = clip.tokenize(self.style_categories).to(self.device)
                    self.style_text_embeddings = self.model.encode_text(self.style_text_features)
                    self.style_text_embeddings /= self.style_text_embeddings.norm(dim=-1, keepdim=True)
                
            except ImportError:
                logger.error("CLIP not installed. Please install with: pip install clip")
                self.enabled = False
            except Exception as e:
                logger.error(f"Error initializing CLIP model: {e}")
                self.enabled = False
                
        elif self.model_name == "vit":
            try:
                from transformers import ViTForImageClassification, ViTImageProcessor
                
                logger.info(f"Loading ViT model for style analysis on {self.device}")
                model_name = "google/vit-base-patch16-224"
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
                logger.info("ViT model loaded successfully")
                
            except ImportError:
                logger.error("Transformers not installed. Please install with: pip install transformers")
                self.enabled = False
            except Exception as e:
                logger.error(f"Error initializing ViT model: {e}")
                self.enabled = False
                
        elif self.model_name == "custom":
            # Custom style analysis implementation
            logger.info("Using custom style analysis implementation")
            
        else:
            logger.warning(f"Unknown style analysis model: {self.model_name}. Style analysis will be disabled.")
            self.enabled = False
    
    def analyze(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the artistic style of an image.
        
        Args:
            image_data: Dictionary containing image data and metadata
            
        Returns:
            Dictionary containing style analysis results:
            {
                'style_embedding': numpy array of style embedding,
                'style_classification': list of (style, confidence) tuples,
                'style_features': dictionary of style-related features
            }
        """
        if not self.enabled:
            logger.warning("Style analysis is disabled. Returning empty results.")
            return {
                'style_embedding': None,
                'style_classification': [],
                'style_features': {}
            }
        
        logger.info(f"Analyzing style for image: {image_data.get('id', 'unknown')}")
        
        # Get image data
        image = None
        if 'image_data' in image_data and image_data['image_data'] is not None:
            # Convert image bytes to PIL Image
            image = Image.open(io.BytesIO(image_data['image_data']))
        elif 'image_path' in image_data and image_data['image_path'] is not None:
            # Load image from path
            image_path = image_data['image_path']
            if os.path.exists(image_path):
                image = Image.open(image_path)
        
        if image is None:
            logger.error("No valid image data found for style analysis")
            return {
                'style_embedding': None,
                'style_classification': [],
                'style_features': {}
            }
        
        # Analyze style based on the configured model
        if self.model_name == "clip":
            return self._analyze_with_clip(image)
        elif self.model_name == "vit":
            return self._analyze_with_vit(image)
        elif self.model_name == "custom":
            return self._analyze_with_custom(image)
        else:
            return {
                'style_embedding': None,
                'style_classification': [],
                'style_features': {}
            }
    
    def _analyze_with_clip(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image style using CLIP.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing style analysis results
        """
        import clip
        
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate image embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with style categories
            similarity = (100.0 * image_features @ self.style_text_embeddings.T).softmax(dim=-1)
            
            # Get top style classifications
            values, indices = similarity[0].topk(5)
            style_classification = [
                (self.style_categories[idx], float(val))
                for val, idx in zip(values, indices)
            ]
        
        # Convert embedding to numpy array
        style_embedding = image_features[0].cpu().numpy()
        
        # Extract style features (color, texture, composition)
        style_features = self._extract_style_features(image)
        
        return {
            'style_embedding': style_embedding,
            'style_classification': style_classification,
            'style_features': style_features
        }
    
    def _analyze_with_vit(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image style using ViT.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing style analysis results
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate image embedding and classification
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get predicted class
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.model.config.id2label[predicted_class_idx]
            
            # Get embedding from last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            style_embedding = last_hidden_state[:, 0].cpu().numpy()  # Use CLS token
        
        # Extract style features
        style_features = self._extract_style_features(image)
        
        return {
            'style_embedding': style_embedding[0],
            'style_classification': [(predicted_class, float(logits.softmax(dim=-1)[0, predicted_class_idx]))],
            'style_features': style_features
        }
    
    def _analyze_with_custom(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image style using custom implementation.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing style analysis results
        """
        # Custom style analysis implementation
        # This is a placeholder for custom implementation
        
        # Extract basic style features
        style_features = self._extract_style_features(image)
        
        return {
            'style_embedding': np.zeros(512),  # Placeholder embedding
            'style_classification': [("Unknown", 1.0)],
            'style_features': style_features
        }
    
    def _extract_style_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract style-related features from an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing style features
        """
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Extract color features
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Calculate average RGB values
            avg_color = img_array.mean(axis=(0, 1))
            
            # Calculate color histogram
            r_hist = np.histogram(img_array[:, :, 0], bins=8, range=(0, 256))[0] / img_array.size
            g_hist = np.histogram(img_array[:, :, 1], bins=8, range=(0, 256))[0] / img_array.size
            b_hist = np.histogram(img_array[:, :, 2], bins=8, range=(0, 256))[0] / img_array.size
            
            color_features = {
                'avg_rgb': avg_color.tolist(),
                'r_hist': r_hist.tolist(),
                'g_hist': g_hist.tolist(),
                'b_hist': b_hist.tolist()
            }
        else:
            # Grayscale or invalid image
            color_features = {
                'avg_rgb': [0, 0, 0],
                'r_hist': [0] * 8,
                'g_hist': [0] * 8,
                'b_hist': [0] * 8
            }
        
        # Extract basic texture features (placeholder)
        texture_features = {
            'contrast': 0.0,
            'energy': 0.0,
            'homogeneity': 0.0
        }
        
        # Extract basic composition features
        composition_features = {
            'aspect_ratio': image.width / image.height if image.height > 0 else 0,
            'width': image.width,
            'height': image.height
        }
        
        return {
            'color': color_features,
            'texture': texture_features,
            'composition': composition_features
        }
    
    def compare_styles(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two style embeddings and return a similarity score.
        
        Args:
            embedding1: First style embedding
            embedding2: Second style embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Normalize to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)


# Factory function to create a style analyzer
def create_style_analyzer(config: Dict[str, Any]) -> StyleAnalyzer:
    """
    Create a style analyzer with the specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured StyleAnalyzer instance
    """
    return StyleAnalyzer(config)


# Function to analyze a collection of images and create a style similarity matrix
def create_style_similarity_matrix(images: List[Dict[str, Any]], analyzer: StyleAnalyzer) -> Dict[str, Any]:
    """
    Analyze a collection of images and create a style similarity matrix.
    
    Args:
        images: List of image dictionaries
        analyzer: StyleAnalyzer instance
        
    Returns:
        Dictionary containing:
        - embeddings: Dictionary mapping image IDs to style embeddings
        - similarity_matrix: 2D array of similarity scores
        - image_ids: List of image IDs in the same order as the similarity matrix
    """
    # Analyze each image
    embeddings = {}
    image_ids = []
    
    for image in images:
        image_id = image.get('id', f"img_{len(embeddings)}")
        image_ids.append(image_id)
        
        # Analyze image style
        style_analysis = analyzer.analyze(image)
        embeddings[image_id] = style_analysis['style_embedding']
    
    # Create similarity matrix
    n = len(image_ids)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = analyzer.compare_styles(
                    embeddings[image_ids[i]], 
                    embeddings[image_ids[j]]
                )
    
    return {
        'embeddings': embeddings,
        'similarity_matrix': similarity_matrix.tolist(),
        'image_ids': image_ids
    }


# Function to find similar images based on style
def find_similar_images(query_image: Dict[str, Any], image_collection: List[Dict[str, Any]], 
                       analyzer: StyleAnalyzer, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find images with similar artistic style to the query image.
    
    Args:
        query_image: Query image dictionary
        image_collection: Collection of image dictionaries to search
        analyzer: StyleAnalyzer instance
        top_k: Number of similar images to return
        
    Returns:
        List of dictionaries containing similar images and similarity scores
    """
    # Analyze query image
    query_analysis = analyzer.analyze(query_image)
    query_embedding = query_analysis['style_embedding']
    
    if query_embedding is None:
        logger.error("Failed to generate embedding for query image")
        return []
    
    # Calculate similarity with each image in the collection
    similarities = []
    
    for image in image_collection:
        image_id = image.get('id', 'unknown')
        
        # Skip the query image if it's in the collection
        if image_id == query_image.get('id', 'query'):
            continue
        
        # Analyze image style
        image_analysis = analyzer.analyze(image)
        image_embedding = image_analysis['style_embedding']
        
        if image_embedding is not None:
            # Calculate similarity
            similarity = analyzer.compare_styles(query_embedding, image_embedding)
            
            similarities.append({
                'image': image,
                'similarity': similarity,
                'style_classification': image_analysis['style_classification']
            })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top-k similar images
    return similarities[:top_k]