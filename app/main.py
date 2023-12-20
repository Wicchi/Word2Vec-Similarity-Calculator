from download_preprocess import DownloadPreprocess
from data_processing import DataProcessor
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process Word2Vec embeddings and calculate similarity.')
    parser.add_argument('word_embeddings_file', type=str, help='Path to the Word2Vec embeddings file')
    parser.add_argument('phrases_file', type=str, help='Path to the phrases file')
    parser.add_argument('--mode', choices=['batch', 'on_the_fly'], default='on_the_fly', help='Execution mode: batch or on-the-fly (default: batch)')
    args = parser.parse_args()

    processor = DataProcessor(word_embeddings_file=args.word_embeddings_file, phrases_file=args.phrases_file)

    if args.mode == 'batch':
        similarity_matrix, top_n_names = processor.calculate_similarity_batch(top_n=5)
        print(f"Similarity Matrix: {similarity_matrix}")
        print("Top N Decoded Names:")
        for i, names in enumerate(top_n_names):
            print(f"Phrase {i + 1}: {', '.join(names)}")
    elif args.mode == 'on_the_fly':
        user_input = input("Enter a phrase: ")
        closest_match, distance, top_n_names = processor.calculate_similarity_on_the_fly(user_input, top_n=5)
        print(f"Closest Match: {closest_match}, Distance: {distance}")


if __name__ == "__main__":
    main()


