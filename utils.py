import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# 加载韩语 spaCy 模型
nlp = spacy.load("ko_core_news_sm")

def sentence_segmentation(text):
    """
    将文本分割成句子。

    Args:
        text (str): 要分割的文本。

    Returns:
        list: 文本中的句子列表。
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def vectorize_tfidf(sentences):
    """
    使用 TF-IDF 将句子向量化。

    Args:
        sentences (list): 要向量化的句子列表。

    Returns:
        list: 句子的 TF-IDF 向量列表。
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.toarray()

def calculate_similarity(sentence_vectors):
    """
    计算句子向量之间的余弦相似度。

    Args:
        sentence_vectors (list): 句子向量列表。

    Returns:
        numpy.ndarray: 句子之间的相似度矩阵。
    """
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix

def apply_textrank(similarity_matrix, sentences):
    """
    应用 TextRank 算法对句子进行排名。

    Args:
        similarity_matrix (numpy.ndarray): 句子之间的相似度矩阵。
        sentences (list): 句子列表。

    Returns:
        list: 排名后的句子列表，每个句子都是一个元组 (排名, 句子)。
    """
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences
