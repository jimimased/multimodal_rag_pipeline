# Multimodal RAG Pipeline

A comprehensive multimodal Retrieval Augmented Generation (RAG) pipeline that can process text, images, audio, and complex PDF layouts. This system extracts, analyzes, and indexes content from multiple modalities to a vectorisation model to enable enhanced information retrieval and aesthetic analysis via LLM in Anthropology & Cultural studies.

# Multimodal RAG Pipeline

## Architecture

```mermaid
flowchart TD
    subgraph Input
        PDF[PDF Documents]
        DOCX[DOCX Documents]
        HTML[HTML Documents]
        Images[Image Files]
        Audio[Audio Files]
    end
    
    subgraph Processing
        DocLoader[Document Loaders]
        LayoutEngine[Layout Analysis Engine]
        TextProc[Text Processor]
        ImageProc[Image Processor]
        AudioProc[Audio Processor]
        Fusion[Multimodal Fusion]
    end
    
    subgraph Embedding
        TextEmbed[Text Embedding Models]
        ImageEmbed[Image Embedding Models]
        AudioEmbed[Audio Embedding Models]
    end
    
    subgraph Storage
        VectorDB[(Vector Database)]
        MetadataDB[(Metadata Store)]
    end
    
    subgraph Retrieval
        QueryProc[Query Processor]
        HybridSearch[Hybrid Search]
        Reranker[Result Reranker]
    end
    
    subgraph Generation
        PromptEng[Prompt Engineering]
        LLM[Large Language Model]
        ResponseGen[Response Generator]
    end
    
    PDF --> DocLoader
    DOCX --> DocLoader
    HTML --> DocLoader
    Images --> DocLoader
    Audio --> DocLoader
    
    DocLoader --> LayoutEngine
    LayoutEngine --> TextProc
    LayoutEngine --> ImageProc
    LayoutEngine --> AudioProc
    
    TextProc --> Fusion
    ImageProc --> Fusion
    AudioProc --> Fusion
    
    Fusion --> TextEmbed
    Fusion --> ImageEmbed
    Fusion --> AudioEmbed
    
    TextEmbed --> VectorDB
    ImageEmbed --> VectorDB
    AudioEmbed --> VectorDB
    
    Fusion --> MetadataDB
    
    UserQuery[User Query] --> QueryProc
    QueryProc --> HybridSearch
    VectorDB --> HybridSearch
    MetadataDB --> HybridSearch
    
    HybridSearch --> Reranker
    Reranker --> PromptEng
    PromptEng --> LLM
    LLM --> ResponseGen
    ResponseGen --> FinalResponse[Final Response]

## Features

### Document Processing
- **Document Ingestion**: Support for PDF, DOCX, HTML, and other formats
- **Metadata Extraction**: File properties and structural information
- **Layout Analysis**: Document structure recognition, table detection, and extraction
- **Modality Extraction**: Text, images, and audio with context preservation

### Content Processing
- **Text Processing**: Semantic chunking, NER, and custom text classifiers
- **Image Analysis**: OCR, image captioning, object detection, and style analysis
- **Multimodal Fusion**: Cross-modal relationship mapping and context preservation

### Embedding and Indexing
- **Text Embeddings**: State-of-the-art models with efficient indexing
- **Image Embeddings**: Vision models with visual similarity indexing
- **Vector Database Integration**: Pinecone, Weaviate, Qdrant, and Chroma support

### Retrieval and Generation
- **Query Understanding**: Multimodal query analysis and formulation
- **Hybrid Retrieval**: Combining BM25 and vector search
- **LLM Integration**: Prompt engineering templates for multimodal RAG

## Architecture

The pipeline is designed with a modular architecture that allows for easy customization and extension. It follows a hybrid VSCode/Google Colab workflow:

1. Design and develop the pipeline architecture in VSCode
2. Implement GPU-intensive components to run in Google Colab
3. Create a modular design that works across both environments

### Pipeline Visualization

The project includes several ways to visualize the pipeline architecture:

1. **Interactive HTML Visualization**: Open `pipeline_visualization.html` in a web browser to see an interactive visualization of the pipeline components.

2. **Mermaid Diagrams**: The `pipeline_architecture.md` file contains Mermaid diagrams that can be viewed in any Markdown viewer that supports Mermaid (like GitHub or VS Code with the Mermaid extension).

3. **Graphviz Diagrams**: You can generate high-quality pipeline diagrams using the provided Python script:

```bash
# Install graphviz (required for the script)
pip install graphviz

# Generate pipeline diagrams
python -m multimodal_rag_pipeline.utils.generate_pipeline_diagram --output diagrams --format png
```

This will generate two diagram files:
- `diagrams/pipeline_diagram.png`: Overall pipeline architecture
- `diagrams/style_analysis_diagram.png`: Detailed view of the artistic style analysis component

## Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/multimodal_rag_pipeline.git
cd multimodal_rag_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for specific features:
```bash
# For OCR support
pip install pytesseract
apt-get install tesseract-ocr

# For PDF processing
pip install pdfplumber pdf2image
apt-get install poppler-utils

# For GPU acceleration
pip install faiss-gpu
```

## Usage

### Basic Usage

```python
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

### Command Line Interface

```bash
# Index documents
python main.py --input_dir /path/to/documents --mode index

# Query the indexed documents
python main.py --mode query --query "Your query here"

# Run the full pipeline (index and query)
python main.py --input_dir /path/to/documents --mode end_to_end
```

### Google Colab Integration

The pipeline includes a Jupyter notebook (`notebooks/multimodal_rag_colab.ipynb`) that demonstrates how to use the pipeline in Google Colab with GPU acceleration. The notebook covers:

1. Setting up the environment
2. Processing multimodal documents
3. Generating embeddings
4. Indexing and retrieval
5. Evaluating performance

## Configuration

The pipeline is configured using a YAML file (`config/config.yaml`). The configuration includes settings for:

- Document processing
- Content processing
- Embedding generation
- Vector database integration
- Retrieval and generation
- Evaluation

You can customize the configuration to suit your specific needs.

## Model Selection

The pipeline supports various models for different components:

- **Text Embeddings**: Models from Sentence Transformers
- **Image Understanding**: CLIP, ViT, or similar vision models
- **Multimodal Embedding**: OpenCLIP, FLAVA, or similar joint models
- **Generation**: GPT-4V, Claude, or similar models with multimodal capabilities

## Performance Optimization

The pipeline includes several performance optimizations:

- Batch processing and parallel execution
- Caching strategies
- Model quantization
- GPU acceleration

## Evaluation Framework

The pipeline includes an evaluation framework for measuring:

- Retrieval quality (precision, recall, NDCG)
- Response quality (ROUGE, BERTScore)
- End-to-end performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their multimodal RAG cookbook components
- [LangChain](https://langchain.com/) for their document processing and retrieval components
- [Sentence Transformers](https://www.sbert.net/) for their embedding models
- [Hugging Face Cookbook: Analyzing Art with HF and FiftyOne](https://huggingface.co/learn/cookbook/en/analyzing_art_with_hf_and_fiftyone) for inspiration on the artistic style analysis component
