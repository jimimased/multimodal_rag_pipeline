# Multimodal RAG Pipeline Architecture

This document provides a visual representation of the Multimodal RAG Pipeline architecture, showing how different components interact to process documents with text, images, audio, and complex layouts.

## High-Level Architecture

```mermaid
graph TD
    subgraph "Document Processing"
        A[Document Ingestion] --> B[Layout Analysis]
        B --> C[Modality Extraction]
    end
    
    subgraph "Content Processing"
        C --> D[Text Processing]
        C --> E[Image Analysis]
        D --> F[Multimodal Fusion]
        E --> F
    end
    
    subgraph "Embedding & Indexing"
        F --> G[Text Embeddings]
        F --> H[Image Embeddings]
        G --> I[Vector Database]
        H --> I
    end
    
    subgraph "Retrieval & Generation"
        J[Query Understanding] --> K[Hybrid Retrieval]
        I --> K
        K --> L[LLM Integration]
        L --> M[Response Generation]
    end
    
    Input[Input Documents] --> A
    Query[User Query] --> J
    M --> Response[Generated Response]
```

## Detailed Component View

```mermaid
graph TD
    classDef processing fill:#f9d5e5,stroke:#333,stroke-width:1px
    classDef embedding fill:#eeeeee,stroke:#333,stroke-width:1px
    classDef retrieval fill:#d5f9e5,stroke:#333,stroke-width:1px
    classDef storage fill:#d5e5f9,stroke:#333,stroke-width:1px
    
    %% Document Processing
    DocLoader[Document Loaders] --> MetaExtract[Metadata Extraction]
    MetaExtract --> LayoutAnalysis[Layout Analysis]
    LayoutAnalysis --> TextExtract[Text Extraction]
    LayoutAnalysis --> ImageExtract[Image Extraction]
    LayoutAnalysis --> AudioExtract[Audio Extraction]
    
    %% Content Processing
    TextExtract --> SemChunk[Semantic Chunking]
    TextExtract --> NER[Named Entity Recognition]
    TextExtract --> TextClass[Text Classification]
    
    ImageExtract --> OCR[OCR]
    ImageExtract --> ImgCaption[Image Captioning]
    ImageExtract --> ObjDetect[Object Detection]
    ImageExtract --> StyleAnalysis[Artistic Style Analysis]
    
    SemChunk --> CrossModal[Cross-Modal Mapping]
    NER --> CrossModal
    TextClass --> CrossModal
    OCR --> CrossModal
    ImgCaption --> CrossModal
    ObjDetect --> CrossModal
    StyleAnalysis --> CrossModal
    
    %% Embedding & Indexing
    CrossModal --> TextEmbed[Text Embedding Models]
    CrossModal --> ImageEmbed[Image Embedding Models]
    
    TextEmbed --> VectorDB[(Vector Database)]
    ImageEmbed --> VectorDB
    
    %% Retrieval & Generation
    Query[User Query] --> QueryAnalysis[Query Analysis]
    QueryAnalysis --> BM25[BM25 Search]
    QueryAnalysis --> VectorSearch[Vector Search]
    
    VectorDB --> VectorSearch
    BM25 --> HybridRank[Hybrid Ranking]
    VectorSearch --> HybridRank
    
    HybridRank --> PromptTemplate[Prompt Templates]
    PromptTemplate --> LLM[Large Language Model]
    LLM --> ResponseVal[Response Validation]
    ResponseVal --> Response[Final Response]
    
    %% Apply classes
    class DocLoader,MetaExtract,LayoutAnalysis,TextExtract,ImageExtract,AudioExtract processing
    class SemChunk,NER,TextClass,OCR,ImgCaption,ObjDetect,StyleAnalysis,CrossModal processing
    class TextEmbed,ImageEmbed embedding
    class VectorDB storage
    class QueryAnalysis,BM25,VectorSearch,HybridRank,PromptTemplate,LLM,ResponseVal retrieval
```

## Artistic Style Analysis Integration

```mermaid
graph TD
    classDef main fill:#f9d5e5,stroke:#333,stroke-width:1px
    classDef models fill:#d5e5f9,stroke:#333,stroke-width:1px
    classDef output fill:#d5f9e5,stroke:#333,stroke-width:1px
    
    Image[Artwork Image] --> StyleAnalyzer[Style Analyzer]
    StyleAnalyzer --> CLIP[CLIP Model]
    StyleAnalyzer --> ViT[ViT Model]
    
    CLIP --> StyleEmbed[Style Embeddings]
    ViT --> StyleEmbed
    
    StyleEmbed --> StyleClass[Style Classification]
    StyleEmbed --> StyleFeatures[Style Features]
    StyleEmbed --> SimilarityMatrix[Style Similarity Matrix]
    
    StyleClass --> RAGIndex[RAG Index]
    StyleFeatures --> RAGIndex
    SimilarityMatrix --> StyleQuery[Style-Based Queries]
    
    StyleQuery --> Response[Generated Response]
    
    class StyleAnalyzer main
    class CLIP,ViT models
    class StyleClass,StyleFeatures,SimilarityMatrix,StyleQuery output
```

## VSCode/Colab Hybrid Workflow

```mermaid
graph LR
    classDef vscode fill:#007acc,color:white,stroke:#333,stroke-width:1px
    classDef colab fill:#F9AB00,color:white,stroke:#333,stroke-width:1px
    classDef shared fill:#cccccc,stroke:#333,stroke-width:1px
    
    VSCode[VSCode Environment] --> Config[Configuration]
    VSCode --> DocProc[Document Processing]
    VSCode --> ContentProc[Content Processing]
    
    Config --> SharedCode[Shared Code Repository]
    DocProc --> SharedCode
    ContentProc --> SharedCode
    
    SharedCode --> Colab[Google Colab]
    Colab --> GPUEmbed[GPU Embedding Generation]
    Colab --> ModelTrain[Model Training/Fine-tuning]
    Colab --> Evaluation[Performance Evaluation]
    
    GPUEmbed --> SharedData[(Shared Data Storage)]
    ModelTrain --> SharedData
    Evaluation --> SharedData
    
    SharedData --> VSCode
    
    class VSCode,DocProc,ContentProc vscode
    class Colab,GPUEmbed,ModelTrain,Evaluation colab
    class SharedCode,SharedData,Config shared
```

## Data Flow Diagram

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