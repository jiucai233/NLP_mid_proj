import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def sentence_segmentation(text, language="en"):
    """
    Segment the text into sentences.

    Args:
        text (str): The text to be segmented.
        language (str): The language of the text ("en" for English, "ko" for Korean, "zh" for Chinese). Defaults to "ko".

    Returns:
        list: A list of sentences in the text.
    """
    if language == "en":
        nlp = spacy.load("en_core_web_sm")
    elif language == "ko":
        nlp = spacy.load("ko_core_news_sm")
    elif language == "zh":
        # TODO: Install and load the Chinese spaCy model (e.g., zh_core_web_sm)
        # nlp = spacy.load("zh_core_web_sm")
        raise NotImplementedError("The Chinese spaCy model is not yet installed.")
    else:
        raise ValueError("Unsupported language. Please select 'en', 'ko' or 'zh'.")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def vectorize_tfidf(sentences):
    """
    Vectorize sentences using TF-IDF.

    Args:
        sentences (list): A list of sentences to be vectorized.

    Returns:
        list: A list of TF-IDF vectors of the sentences.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.toarray()
def apply_textrank(similarity_matrix, sentences):
    """
    Apply the TextRank algorithm to rank sentences.

    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix between sentences.
        sentences (list): List of sentences.

    Returns:
        list: A list of ranked sentences, each sentence is a tuple (rank, sentence).
    """
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences
