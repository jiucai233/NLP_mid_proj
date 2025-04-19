from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from english_utils import sentence_segmentation_en, vectorize_tfidf_en, apply_textrank_en
from korean_utils import sentence_segmentation_ko, vectorize_tfidf_ko, apply_textrank_ko

def calling_data(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text
def summarize_en(text, num_sentences=5, reference_summary=None):
    """
    从英语足球比赛新闻文章中提取关键句子并生成摘要。

    Args:
        text (str): 新闻文章的文本。
        num_sentences (int): 摘要中要包含的句子数量。

    Returns:
        str: 新闻文章的摘要。
    """

    # 1. 句子分割
    sentences = sentence_segmentation_en(text)

    # 2. 句子向量化 (TF-IDF)
    sentence_vectors = vectorize_tfidf_en(sentences)

    # 3. 构建相似度矩阵
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. 应用 TextRank 算法
    ranked_sentences = apply_textrank_en(similarity_matrix, sentences)


    # 5. 选择排名最高的句子
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    summary = " ".join(summary_sentences)
    summary_vectors = vectorize_tfidf_en([summary])

    cosine_sim = cosine_similarity(sentence_vectors, summary_vectors)
    
    # Calculate loss (1 - similarity as a simple loss function)
    # Lower loss is better
    loss = 1 - cosine_sim
    
    # If reference summary is provided, also calculate similarity with it
    ref_sim = None
    if reference_summary:
        ref_sim = cosine_similarity(reference_summary, summary_vectors)
    
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
    article_text_en = calling_data("data\en\en1")
    summary_en = summarize_en(article_text_en, num_sentences=2)
    print("English Summary:", summary_en)