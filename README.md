# Project
## Overview

This project is designed to process Word2Vec embeddings and calculate similarity between phrases. It supports both batch processing and on-the-fly user input. The provided scripts download and preprocess Word2Vec embeddings, and then utilize them to find similar phrases based on cosine similarity.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python (>=3.6)
- Pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Wicchi/Galytix_tech_task.git
   cd Galytix_tech_tas
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Batch Processing

To perform batch processing, use the following command:

```bash
python main.py <word_embeddings_file> <phrases_file> --mode batch
```

Replace `<word_embeddings_file>` with the path to your Word2Vec embeddings file and `<phrases_file>` with the path to your phrases file.

### On-the-Fly Processing

For on-the-fly processing, run:

```bash
python main.py <word_embeddings_file> <phrases_file> --mode on_the_fly
```

You will be prompted to enter a phrase, and the system will find the closest match based on the provided input.

## Project Structure

- **main.py**: Main script for executing batch or on-the-fly processing.
- **data_processing.py**: Module containing the `DataProcessor` class for calculating similarity.
- **download_preprocess.py**: Module for downloading and preprocessing Word2Vec embeddings.
- **requirements.txt**: List of project dependencies.
- **vectors.csv**: Flat file containing Word2Vec embeddings after preprocessing.

## Dependencies

- gensim
- pandas
- numpy
- scipy
- nltk
- requests

## Contributing

Feel free to contribute by opening issues or pull requests. All contributions are welcome!

---

Adjust the sections and content according to your project's specific details. The example assumes a simple project structure and provides basic usage instructions.
