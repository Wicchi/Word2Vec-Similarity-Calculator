import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class DataProcessor:
    def __init__(self, word_embeddings_file, phrases_file):
        print(f"Word Embeddings File Path: {word_embeddings_file}")
        self.word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=True, limit=1000000)
        self.phrases_df = pd.read_csv(phrases_file, encoding='latin-1')
        self.stemmer = PorterStemmer()

    def preprocess_phrases(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def calculate_similarity_batch(self, top_n=5):
        # Batch execution: Calculate L2 distance (Euclidean distance) or Cosine distance
        # of each phrase to all other phrases and store results

        # Assuming phrases_df has a column 'Phrases' containing the phrases
        phrase_embeddings = self.phrases_df['Phrases'].apply(self.preprocess_phrases) \
                                                .apply(lambda x: self.calculate_embedding(x))

        # Example: Calculate Cosine Similarity
        similarity_matrix = 1 - cdist(np.stack(phrase_embeddings), np.stack(phrase_embeddings), metric='cosine')

        # Get the top N similar phrases for each phrase
        top_n_indices = np.argsort(similarity_matrix, axis=1)[:, :-top_n-1:-1]
        top_n_names = [
            [self.phrases_df.loc[idx, 'Phrases'] for idx in row]
            for row in top_n_indices
        ]

        # Store or return the similarity_matrix and top N names as needed
        return similarity_matrix, top_n_names

    def preprocess_phrases_column(self):
        self.phrases_df['Phrases'] = self.phrases_df['Phrases'].apply(
            lambda x: self.calculate_embedding(x.split()) if x else np.zeros(300, dtype=np.float32)
        )

    def calculate_embedding(self, words):
        valid_embeddings = []

        for word in words:
            stemmed_word = self.stemmer.stem(word)

            try:
                valid_embeddings.append(self.word_embeddings[stemmed_word])
            except KeyError:
                valid_embeddings.append(np.zeros_like(self.word_embeddings['word']))

        embedding = np.mean(valid_embeddings, axis=0)
        return embedding / np.linalg.norm(embedding)

    def preprocess_user_input(self, user_input):
        words = user_input.split()[:50]
        words.extend(['<PAD>' for _ in range(50 - len(words))])
        return ' '.join(words)
    
    def decode_names(self, encoded_names):
        decoded_names = [self.phrases_df.loc[idx, 'Phrases'] for idx in encoded_names]
        return decoded_names

    def calculate_similarity_on_the_fly(self, user_input, top_n=5):
        user_input_embedding = self.calculate_embedding(self.preprocess_user_input(user_input))

        self.preprocess_phrases_column()
        phrases_embeddings = np.vstack(self.phrases_df['Phrases'].values)

        if user_input_embedding.shape != phrases_embeddings.shape[1:]:
            user_input_embedding = user_input_embedding.reshape(1, -1)

        user_input_embedding = user_input_embedding / np.linalg.norm(user_input_embedding)

        similarity_scores = np.dot(phrases_embeddings, user_input_embedding) / (
            np.linalg.norm(phrases_embeddings, axis=1) * np.linalg.norm(user_input_embedding)
        )

        # Return the closest match, distance, and decoded top N names
        closest_match_index = np.argmax(similarity_scores)
        closest_match = self.phrases_df.loc[closest_match_index, 'Phrases']
        distance = similarity_scores[closest_match_index]

        return closest_match, distance
   


    def get_top_n_names(self, phrase_index, n=5):
        # Get the similarity scores for the specified phrase_index
        similarity_scores = self.similarity_matrix[phrase_index]

        # Find the indices of the top N similarity scores
        top_n_indices = np.argsort(similarity_scores)[::-1][:n]

        # Retrieve the names corresponding to the top N indices
        top_n_names = self.phrases_df.iloc[top_n_indices]['Phrases'].tolist()

        return top_n_names
    
    def calculate_similarity(self, execution_mode):
        if execution_mode == 'batch':
            similarity_matrix = self.calculate_similarity_batch()
            print(f"Similarity Matrix: {similarity_matrix}")
        elif execution_mode == 'on_the_fly':
            user_input = input("Enter a phrase: ")
            closest_match, distance = self.calculate_similarity_on_the_fly(user_input)
            print(f"Closest Match: {closest_match}, Distance: {distance}")
        else:
            print("Invalid execution mode. Please choose 'batch' or 'on_the_fly'.")
