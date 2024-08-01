import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find
import re

def ensure_nltk_resources():
    try:
        find('tokenizers/punkt')
        find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

class Preprocessing:

    def __init__(self) -> None:
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.contractions = {
            "he's": "he is",
            "i've": "i have",
            "can't": "cannot",
            "they're": "they are",
            "let's": "let us",
            "it's": "it is"
        }
        self.words_to_numbers = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'zero': '0'
        }
        self.contraction_pattern = re.compile(r'\b(' + '|'.join(self.contractions.keys()) + r')\b')
        self.number_pattern = re.compile(r'\b(' + '|'.join(self.words_to_numbers.keys()) + r')\b')

    def process_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = self.contraction_pattern.sub(lambda x: self.contractions[x.group()], text)  # Expand contractions
        text = re.sub(r'https://\S+|http://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
        text = self.number_pattern.sub(lambda x: self.words_to_numbers[x.group()], text)  # Convert number-words
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        words = [word for word in word_tokenize(text) if word not in self.stop_words]
        return ' '.join(words)

    def process(self, file_path='news.csv', nrows=10, skiprows=[]):
        print("Process method started")

        try:
            df = pd.read_csv(file_path, skiprows=skiprows, nrows=nrows)
            if 'Headline' not in df.columns:
                raise ValueError("The 'Headline' column is missing in the CSV file.")
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            return []

        raw = df['Headline'].tolist()
        print(f"Raw Headlines: {raw}")

        if not raw:
            print("No headlines found in the data.")
            return []

        corpus = [self.process_text(text) for text in raw]

        print(f"Final Corpus: {corpus}")
        return corpus

if __name__ == "__main__":
    preprocessor = Preprocessing()
    corpus = preprocessor.process(file_path='news.csv', nrows=10)
    print("Processed Corpus:", corpus)
