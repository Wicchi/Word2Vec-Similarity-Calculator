# Word2Vec Similarity Calculator

This project allows you to calculate semantic similarity between phrases using Word2Vec embeddings. It provides two execution modes: batch and on-the-fly.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

- Docker installed on your machine
- Python 3.8

### Installing

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/word2vec-similarity.git
    cd word2vec-similarity
    ```

2. Build the Docker image:

    ```bash
    docker build -t word2vec_project .
    ```

3. Run the Docker container:

    ```bash
    docker run -p 4000:80 word2vec_project
    ```

## Usage

The project supports two execution modes:

1. **Batch Mode:**
   - Calculates similarity using a batch approach.
   - Run the following command:

        ```bash
        docker exec -it <container_id> python main.py --mode batch
        ```

2. **On-the-Fly Mode:**
   - Allows you to input a phrase and find the closest match.
   - Run the following command:

        ```bash
        docker exec -it <container_id> python main.py --mode on_the_fly
        ```

## Configuration

You can customize the project's behavior using command-line options:

- `--mode`: Choose between 'batch' and 'on_the_fly'.
- `--use_tf`: Use TensorFlow Word2Vec.
- `word_embeddings_file`: Path to the Word2Vec embeddings file.
- `phrases_file`: Path to the phrases file.

Example:

```bash
docker exec -it <container_id> python main.py --mode batch --use_tf --word_embeddings_file data/GoogleNews-vectors-negative300.bin.gz --phrases_file /data/phrases.csv
```


Based on task from Galytix
