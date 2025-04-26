import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt_tab')
nltk.download('stopwords')
def preprocess_text(text):
    """
    Preprocesses the given text by performing tokenization, lowercasing,
    stop word removal, and punctuation removal.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """

    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Stop word removal
    #stop_words = set(stopwords.words('english'))
    #tokens = [token for token in tokens if token not in stop_words]

    # Punctuation removal
    tokens = [token for token in tokens if token not in string.punctuation]

    return tokens
    # 文本预处理
def build_vocabulary(tokenized_sentences):
    # 모든 단어 수집
    all_words = [word for sentence in tokenized_sentences for word in sentence]
    
    # 단어 빈도 계산
    word_counts = Counter(all_words)
    
    # 빈도순으로 정렬 (가장 빈번한 것부터)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 단어에 ID 할당 (빈도순)
    word_to_id = {word: i for i, (word, _) in enumerate(sorted_words)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    return word_to_id, id_to_word, word_counts

def read_abstract_from_pkl(path):
    """
    read abstract from pkl file.

    Args:
        path (str): pkl file path.

    Returns:
        str: abstract text.
    """
    df = pd.read_pickle(path)
    abstract_text = df['Abstract'].tolist()
    return abstract_text

if __name__ == '__main__':
    text = "This is an example sentence with some punctuation and stop words."
    preprocessed_text = preprocess_text(text)
    print(f"Original text: {text}")
    print(f"Preprocessed text: {preprocessed_text}")
