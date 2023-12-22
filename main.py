from download_preprocess import DownloadPreprocess
from data_processing import DataProcessor
import argparse
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    # Specify the paths for the word embeddings and phrases files
    word_embeddings_file = 'data/GoogleNews-vectors-negative300.bin.gz'  # Updated path
    phrases_file = 'data/phrases.csv'  # New path

    # Check if the word embeddings file exists; if not, download and preprocess
    if not os.path.isfile(word_embeddings_file):
        download_preprocess = DownloadPreprocess(
            url='https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM',
            output_file=word_embeddings_file
        )
        download_preprocess.download_and_preprocess()

    # Create an ArgumentParser for additional options
    parser = argparse.ArgumentParser(description='Process Word2Vec embeddings and calculate similarity.')
    parser.add_argument('--mode', choices=['batch', 'on_the_fly'], default='on_the_fly', help='Execution mode: batch or on-the-fly (default: batch)')
    parser.add_argument('word_embeddings_file', default=word_embeddings_file, type=str, help='Path to the Word2Vec embeddings file')
    parser.add_argument('phrases_file', default=phrases_file, type=str, help='Path to the phrases file')
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Error: {e}")
        return

    # Initialize logger
    logger = logging.getLogger(__name__)

    # Log the input arguments
    logger.info("Word Embeddings File: %s", args.word_embeddings_file)
    logger.info("Phrases File: %s", args.phrases_file)  # Corrected attribute name
    logger.info("Execution Mode: %s", args.mode)

    # Initialize DataProcessor with the specified files
    processor = DataProcessor(word_embeddings_file=args.word_embeddings_file, phrases_file=args.phrases_file)

    if args.mode == 'batch':
        similarity_matrix, top_n_names = processor.calculate_similarity_batch(top_n=5)
        logger.info("Similarity Matrix: %s", similarity_matrix)
        logger.info("Top N Decoded Names:")
        for i, names in enumerate(top_n_names):
            logger.info("Phrase %d: %s", i + 1, ', '.join(names))
    elif args.mode == 'on_the_fly':
        user_input = input("Enter a phrase: ")
        closest_match, distance = processor.calculate_similarity_on_the_fly(user_input)
        logger.info("Closest Match: %s, Distance: %s", closest_match, distance)


if __name__ == "__main__":
    main()
