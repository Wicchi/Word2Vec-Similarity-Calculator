from setuptools import setup, find_packages

setup(
    name='word_similarity_app',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'numpy',
        'pandas',
        'scipy',
        'nltk',
        'requests',
    ],
)
