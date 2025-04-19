#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Loader

This module provides functionality to load and validate configuration files
for the multimodal RAG pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration settings
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration format: {config_path}")
        
        # Validate configuration
        validate_config(config)
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from e
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check for required top-level sections
    required_sections = [
        "document_processing",
        "content_processing",
        "embedding",
        "vector_db",
        "retrieval"
    ]
    
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing required configuration section: {section}")
    
    # Validate document processing configuration
    if "document_processing" in config:
        doc_proc = config["document_processing"]
        
        # Check for required document processing subsections
        doc_proc_sections = [
            "document_loaders",
            "layout_analysis",
            "modality_extraction"
        ]
        
        for section in doc_proc_sections:
            if section not in doc_proc:
                logger.warning(f"Missing document processing section: {section}")
    
    # Validate content processing configuration
    if "content_processing" in config:
        content_proc = config["content_processing"]
        
        # Check for required content processing subsections
        content_proc_sections = [
            "text_processing",
            "image_analysis",
            "multimodal_fusion"
        ]
        
        for section in content_proc_sections:
            if section not in content_proc:
                logger.warning(f"Missing content processing section: {section}")
    
    # Validate embedding configuration
    if "embedding" in config:
        embedding = config["embedding"]
        
        # Check for required embedding subsections
        embedding_sections = [
            "text",
            "image"
        ]
        
        for section in embedding_sections:
            if section not in embedding:
                logger.warning(f"Missing embedding section: {section}")
    
    # Validate vector database configuration
    if "vector_db" in config:
        vector_db = config["vector_db"]
        
        # Check for required vector database settings
        if "provider" not in vector_db:
            logger.warning("Missing vector database provider")
    
    # Validate retrieval configuration
    if "retrieval" in config:
        retrieval = config["retrieval"]
        
        # Check for required retrieval subsections
        retrieval_sections = [
            "query",
            "llm"
        ]
        
        for section in retrieval_sections:
            if section not in retrieval:
                logger.warning(f"Missing retrieval section: {section}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add the value
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Dictionary containing the default configuration settings
    """
    # Get the directory of this file
    current_dir = Path(__file__).parent
    
    # Default config is in the config directory
    default_config_path = current_dir.parent.parent / "config" / "config.yaml"
    
    if default_config_path.exists():
        return load_config(default_config_path)
    else:
        logger.warning(f"Default configuration file not found: {default_config_path}")
        return {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
        
    Raises:
        ValueError: If the configuration is invalid
    """
    config_path = Path(config_path)
    
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving configuration to {config_path}")
    
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        logger.info(f"Configuration saved successfully to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise