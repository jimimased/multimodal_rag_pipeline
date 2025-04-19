# Multimodal RAG Pipeline Examples

This directory contains example scripts that demonstrate how to use various components of the Multimodal RAG Pipeline.

## Available Examples

### Artistic Style Analysis

The `artistic_style_analysis_example.py` script demonstrates how to use the artistic style analysis component of the pipeline to:

1. Analyze artistic styles in images
2. Create style similarity matrices
3. Find images with similar artistic styles
4. Integrate style analysis into the RAG pipeline

This example is inspired by the [Hugging Face cookbook article on analyzing art with Hugging Face and FiftyOne](https://huggingface.co/learn/cookbook/en/analyzing_art_with_hf_and_fiftyone).

#### Running the Example

```bash
# Install required dependencies
pip install requests pillow matplotlib numpy

# Option 1: Download sample images from the web
python artistic_style_analysis_example.py --output output_directory --use-web

# Option 2: Load images from Google Drive (for Google Colab)
python artistic_style_analysis_example.py --output output_directory --gdrive /content/drive/MyDrive/SUMBA

# Use GPU acceleration if available
python artistic_style_analysis_example.py --output output_directory --gdrive /content/drive/MyDrive/SUMBA --gpu
```

The script will:
1. Load artwork images (either from Google Drive or by downloading from the web)
2. Analyze the artistic style of each image
3. Create a style similarity matrix
4. Find artworks with similar styles to a query image
5. Demonstrate how to integrate style analysis into the RAG pipeline

#### Google Drive Integration

When using the `--gdrive` option, the script will:

1. Check if running in Google Colab and mount Google Drive if needed
2. Load images from the specified Google Drive directory
3. Extract metadata (artist, title, style) from filenames and directory structure
4. Process the images using the style analyzer

For best results with Google Drive integration:
- Organize your images in subdirectories named after artistic styles (e.g., "Impressionism", "Cubism")
- Name your files with the format "Artist_Title.jpg" (e.g., "VanGogh_StarryNight.jpg")

## Creating Your Own Examples

You can use these examples as templates for creating your own examples that demonstrate specific aspects of the Multimodal RAG Pipeline. To create a new example:

1. Create a new Python script in the `examples` directory
2. Import the necessary components from the pipeline
3. Implement your example logic
4. Add documentation to explain what the example demonstrates

## Integration with Notebooks

For more interactive examples, check out the Jupyter notebooks in the `notebooks` directory:

- `multimodal_rag_colab.ipynb`: Demonstrates the full pipeline in Google Colab with GPU acceleration
- `artistic_style_analysis.ipynb`: Interactive notebook for analyzing artistic styles