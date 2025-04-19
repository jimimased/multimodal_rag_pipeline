# Getting Started with the Multimodal RAG Pipeline

This guide will help you get started with the Multimodal RAG Pipeline, a comprehensive system for processing documents containing text, images, audio, and complex PDF layouts.

## Hybrid VSCode/Google Colab Workflow

The pipeline is designed to support a hybrid workflow:
1. Design and develop the pipeline architecture in VSCode
2. Implement GPU-intensive components to run in Google Colab
3. Create a modular design that works across both environments

## Installation

### Option 1: Install from the current directory

```bash
# Install the package in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[all]"

# Or install with specific component dependencies
pip install -e ".[pdf,image,vector_db]"
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/jimimased/multimodal_rag_pipeline.git
```

## Basic Usage

### Command Line Interface

The package provides a command-line interface for easy use:

```bash
# Index documents
multimodal-rag --input_dir /path/to/documents --mode index

# Query the indexed documents
multimodal-rag --mode query --query "Your query here"

# Run the full pipeline (index and query)
multimodal-rag --input_dir /path/to/documents --mode end_to_end
```

### Python API

You can also use the Python API for more flexibility:

```python
from multimodal_rag_pipeline.document_processing import load_documents
from multimodal_rag_pipeline.content_processing.text_processing import process_text
from multimodal_rag_pipeline.content_processing.image_analysis import analyze_images
from multimodal_rag_pipeline.content_processing.multimodal_fusion import fuse_modalities
from multimodal_rag_pipeline.embedding_indexing.text_embeddings import generate_text_embeddings
from multimodal_rag_pipeline.embedding_indexing.image_embeddings import generate_image_embeddings
from multimodal_rag_pipeline.embedding_indexing.vector_db import index_embeddings
from multimodal_rag_pipeline.retrieval_generation.query_understanding import process_query
from multimodal_rag_pipeline.retrieval_generation.llm_integration import generate_response
from multimodal_rag_pipeline.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Process documents
documents = load_documents('input_dir', config["document_processing"])

# Process content
processed_text = process_text(documents, config["content_processing"]["text_processing"])
processed_images = analyze_images(documents, config["content_processing"]["image_analysis"])
fused_content = fuse_modalities(processed_text, processed_images, config["content_processing"]["multimodal_fusion"])

# Generate embeddings
text_embeddings = generate_text_embeddings(processed_text, config["embedding"]["text"])
image_embeddings = generate_image_embeddings(processed_images, config["embedding"]["image"])

# Index embeddings
index_embeddings(text_embeddings, image_embeddings, fused_content, config["vector_db"])

# Process query and generate response
processed_query = process_query("Your query here", config["retrieval"]["query"])
response = generate_response(processed_query, config["retrieval"]["llm"])
print(response)
```

## Google Colab Integration

For GPU-intensive tasks, you can use the provided Jupyter notebook in Google Colab:

1. Upload the `notebooks/multimodal_rag_colab.ipynb` file to Google Colab
2. Follow the instructions in the notebook to set up the environment
3. Run the cells to process documents, generate embeddings, and query the system

## Next Steps

### Google Drive Integration

The pipeline supports loading data from Google Drive when running in Google Colab:

```python
# In a Google Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Load images from Google Drive
from multimodal_rag_pipeline.examples.artistic_style_analysis_example import load_images_from_gdrive

# Path to your Google Drive folder containing images
gdrive_path = "/content/drive/MyDrive/SUMBA"  # Change to your folder path
images = load_images_from_gdrive(gdrive_path)

# Process the images
# ...
```

You can also use the provided example script with Google Drive integration:

```bash
# Run the artistic style analysis example with Google Drive
python -m multimodal_rag_pipeline.examples.artistic_style_analysis_example --gdrive /content/drive/MyDrive/SUMBA
```

### Implementing Missing Components

The current implementation includes the basic structure and some key components, but you'll need to implement the following:

1. **Content Processing Components**:
   - Text processing (semantic chunking, NER, classification)
   - Image analysis (OCR, captioning, object detection)
   - Multimodal fusion (cross-modal relationship mapping)

2. **Embedding and Indexing Components**:
   - Text embeddings generation
   - Image embeddings generation
   - Vector database integration

3. **Retrieval and Generation Components**:
   - Query understanding and processing
   - LLM integration for response generation

### Customizing the Pipeline

You can customize the pipeline by:

1. Modifying the configuration in `config/config.yaml`
2. Implementing custom components for specific use cases
3. Extending the existing components with additional functionality

### Testing and Evaluation

To evaluate the performance of your pipeline:

1. Create a test dataset with ground truth
2. Implement evaluation metrics in the `evaluation` directory
3. Run benchmarks to measure retrieval and generation quality

## Project Structure

```
multimodal_rag_pipeline/
├── config/                  # Configuration files
├── content_processing/      # Content processing components
│   ├── image_analysis/      # Image analysis components
│   ├── multimodal_fusion/   # Multimodal fusion components
│   └── text_processing/     # Text processing components
├── document_processing/     # Document processing components
│   ├── document_loaders/    # Document loaders for various formats
│   ├── layout_analysis/     # Layout analysis components
│   ├── metadata_extraction/ # Metadata extraction components
│   └── modality_extraction/ # Modality extraction components
├── embedding_indexing/      # Embedding and indexing components
│   ├── image_embeddings/    # Image embedding components
│   ├── text_embeddings/     # Text embedding components
│   └── vector_db/          # Vector database integration
├── evaluation/              # Evaluation framework
│   ├── benchmarks/          # Benchmarking tools
│   └── metrics/             # Evaluation metrics
├── notebooks/               # Jupyter notebooks for Colab
├── retrieval_generation/    # Retrieval and generation components
│   ├── llm_integration/     # LLM integration components
│   └── query_understanding/ # Query understanding components
└── utils/                   # Utility functions and classes
```

## Resources

- [Hugging Face Multimodal RAG Cookbook](https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms)
- [Multimodal RAG Patterns](https://vectorize.io/multimodal-rag-patterns/)
- [Analyzing Art with Hugging Face and FiftyOne](https://huggingface.co/learn/cookbook/en/analyzing_art_with_hf_and_fiftyone)