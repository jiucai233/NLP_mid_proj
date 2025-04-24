from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import sentence_segmentation, vectorize_tfidf, apply_textrank
import pandas as pd

def read_abstract_from_pkl(path):
    """
    read abstract from pkl file.

    Args:
        path (str): pkl file path.

    Returns:
        str: abstract text.
    """
    df = pd.read_pickle(path)
    abstract_text = df['Abstract'].str.cat(sep=' ')
    return abstract_text

def summarize_en(text, num_sentences=5, reference_summary=None):
    """
    extractive summarization of English articles.

    Args:
        text (str): English article text.
        num_sentences (int): number of sentences in the summary.
        reference_summary (str): reference summary for comparison.

    Returns:
        str: core sentence of the article.
        metrics (dict): evaluation metrics for the summary.
    """

    # 1. segment sentences
    sentences = sentence_segmentation(text,language='en')

    # 2. vectorize sentences (TF-IDF)
    sentence_vectors = vectorize_tfidf(sentences)

    # 3. construct similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. apply TextRank alghorithm
    ranked_sentences = apply_textrank(similarity_matrix, sentences)

    # 5. choose top ranked sentences
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    summary = " ".join(summary_sentences)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text, summary])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Calculate loss (1 - similarity as a simple loss function)
    # Lower loss is better
    loss = 1 - cosine_sim
    
    # If reference summary is provided, also calculate similarity with it
    ref_sim = None
    if reference_summary:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([reference_summary, summary])
        ref_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    metrics = {
        'cosine_similarity': cosine_sim,
        'loss': loss,
        'reference_similarity': ref_sim
    }
    return summary, metrics

def summarize_ko(text, num_sentences=5):
    """
    从韩语足球比赛新闻文章中提取关键句子并生成摘要。

    Args:
        text (str): 新闻文章的文本。
        num_sentences (int): 摘要中要包含的句子数量。

    Returns:
        str: 新闻文章的摘要。
    """

    # 1. 句子分割
    sentences = sentence_segmentation_ko(text)

    # 2. 句子向量化 (TF-IDF)
    sentence_vectors = vectorize_tfidf_ko(sentences)
    print(f'{sentence_vectors}')
    # 3. 构建相似度矩阵
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. 应用 TextRank 算法
    ranked_sentences = apply_textrank_ko(similarity_matrix, sentences)
    print(f'{ranked_sentences}')

    # 5. 选择排名最高的句子
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    # 6. 生成摘要
    summary = " ".join(summary_sentences)
    return summary

if __name__ == '__main__':
    article_text_en = read_abstract_from_pkl("data/arxiv_papers.pkl")
    summary_en, metrics= summarize_en(article_text_en, num_sentences=2)
    print("English Summary:", summary_en)
    print(f'{metrics=}')
