#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimodal_rag_pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive multimodal RAG pipeline for processing documents with text, images, and audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jimimased/multimodal_rag_pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pypdf>=3.15.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.0.300",
        "faiss-cpu>=1.7.4",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "pdf": [
            "pdfplumber>=0.10.0",
            "pymupdf>=1.22.0",
            "pdf2image>=1.16.0",
            "pytesseract>=0.3.10",
        ],
        "image": [
            "opencv-python>=4.8.0",
            "easyocr>=1.7.0",
            "clip>=1.0",
            "torchvision>=0.15.0",
        ],
        "layout": [
            "layoutparser>=0.3.4",
            "detectron2>=0.6",
        ],
        "vector_db": [
            "pinecone-client>=2.2.0",
            "weaviate-client>=3.20.0",
            "qdrant-client>=1.5.0",
            "chromadb>=0.4.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "bitsandbytes>=0.41.0",
        ],
        "all": [
            "pdfplumber>=0.10.0",
            "pymupdf>=1.22.0",
            "pdf2image>=1.16.0",
            "pytesseract>=0.3.10",
            "opencv-python>=4.8.0",
            "easyocr>=1.7.0",
            "clip>=1.0",
            "torchvision>=0.15.0",
            "layoutparser>=0.3.4",
            "detectron2>=0.6",
            "pinecone-client>=2.2.0",
            "weaviate-client>=3.20.0",
            "qdrant-client>=1.5.0",
            "chromadb>=0.4.0",
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-rag=multimodal_rag_pipeline.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multimodal_rag_pipeline": ["config/*.yaml"],
    },
)