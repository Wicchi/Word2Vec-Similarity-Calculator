import requests
from pathlib import Path
import gensim.downloader as api
from gensim.models import KeyedVectors
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D

class DownloadPreprocess:
    def __init__(self, url, output_file):
        self.url = url
        self.output_file = output_file

    def download_word2vec_vectors(self):
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

            with open(self.output_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error during download: {e}")

    def preprocess_word2vec_vectors(self, use_tf=False):
        if use_tf:
            # Train Word2Vec using a simple neural network in TensorFlow
            corpus = api.load('text8')
            sentences = [sentence for sentence in corpus]

            model = Sequential([
                Embedding(input_dim=len(corpus), output_dim=300, input_length=50),
                GlobalAveragePooling1D()
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(sentences, sentences, epochs=10)

            # Save the trained model
            model.save('word2vec_tf_model.h5')

        else:
            api.load(self.url)

            # Load the Word2Vec vectors directly
            wv = KeyedVectors.load_word2vec_format(self.output_file, binary=True, limit=1000000)

            # Save the vectors as a flat file
            wv.save_word2vec_format('vectors.csv')

    def download_and_preprocess(self, use_tf=False):
        try:
            if not Path(self.output_file).is_file():
                print("Downloading Word2Vec vectors...")
                self.download_word2vec_vectors()
                print("Download complete.")

                print("Preprocessing Word2Vec vectors...")
                self.preprocess_word2vec_vectors(use_tf=use_tf)
                print("Preprocessing complete.")
            else:
                print(f"File '{self.output_file}' already exists. Skipping download and preprocess.")
        except Exception as e:
            print(f"Error: {e}")
