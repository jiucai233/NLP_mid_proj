from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TFIDF_utils import sentence_segmentation, vectorize_tfidf, apply_textrank
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
    return df[['Abstract', 'Title']]

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
    sentences = sentence_segmentation(text, language='en')
    sentences = [s for s in sentences if len(s.strip().split()) > 3]  # remove very short sentences

    if len(sentences) < num_sentences:
        num_sentences = len(sentences)

    # 2. vectorize sentences (TF-IDF)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    # 3. construct similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. apply TextRank algorithm
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)

    # 5. choose top ranked sentences
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: sentences.index(x[1]))
    summary_sentences = [s[1] for s in top_sentences]
    summary = " ".join(summary_sentences)

    # 6. Evaluation metrics
    full_vector = vectorizer.transform([text])
    summary_vector = vectorizer.transform([summary])

    cosine_sim = cosine_similarity(full_vector, summary_vector)[0][0]
    loss = 1 - cosine_sim

    # sentence-level analysis
    summary_indices = [sentences.index(s) for s in summary_sentences]
    other_indices = [i for i in range(len(sentences)) if i not in summary_indices]

    avg_self_similarity = cosine_sim

    if other_indices:
        summary_vecs = sentence_vectors[summary_indices].toarray()
        other_vecs = sentence_vectors[other_indices].toarray()
        avg_diff = 1 - cosine_similarity(summary_vecs.mean(axis=0).reshape(1, -1), other_vecs).mean()
    else:
        avg_diff = None

    ref_sim = None
    if reference_summary:
        ref_vectors = vectorizer.transform([reference_summary])
        ref_sim = cosine_similarity(ref_vectors, summary_vector)[0][0]

    metrics = {
        'cosine_similarity': cosine_sim,
        'loss': loss,
        'average_self_similarity': avg_self_similarity,
        'average_diff_with_others': avg_diff,
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
    sentences = sentence_segmentation(text)

    # 2. 句子向量化 (TF-IDF)
    sentence_vectors = vectorize_tfidf(sentences)
    print(f'{sentence_vectors}')
    # 3. 构建相似度矩阵
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. 应用 TextRank 算法
    ranked_sentences = apply_textrank(similarity_matrix, sentences)
    print(f'{ranked_sentences}')

    # 5. 选择排名最高的句子
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    # 6. 生成摘要
    summary = " ".join(summary_sentences)
    return summary

def find_similar_abstract(input_text, df):
    """
    Finds the most similar abstract in the dataframe to the input text.

    Args:
        input_text (str): The input text.
        df (pd.DataFrame): The dataframe containing abstracts and titles.

    Returns:
        tuple: The most similar abstract and its title.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Abstract'])

    input_vector = tfidf_vectorizer.transform([input_text])

    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)

    most_similar_index = cosine_similarities.argmax()

    abstract = df['Abstract'][most_similar_index]
    title = df['Title'][most_similar_index]
    summary, metrics = summarize_en(abstract, num_sentences=2)

    return summary, title, metrics

if __name__ == '__main__':
    df = read_abstract_from_pkl("data/arxiv_papers.pkl")

    input_text = input("Enter some text: ")

    most_similar_abstract, title, metrics = find_similar_abstract(input_text, df)

    print("Most similar abstract:", most_similar_abstract)
    print("Title:", title)
    print("Metrics:", metrics)
