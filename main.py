from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from english_utils import sentence_segmentation_en, vectorize_tfidf_en, calculate_similarity_en, apply_textrank_en
from korean_utils import sentence_segmentation_ko, vectorize_tfidf_ko, calculate_similarity_ko, apply_textrank_ko

def calling_data(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text
def summarize_en(text, num_sentences=5):
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
    similarity_matrix = calculate_similarity_en(sentence_vectors)

    # 4. 应用 TextRank 算法
    ranked_sentences = apply_textrank_en(similarity_matrix, sentences)

    # 5. 选择排名最高的句子
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    # 6. 生成摘要
    summary = " ".join(summary_sentences)
    return summary

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

    # 3. 构建相似度矩阵
    similarity_matrix = calculate_similarity_ko(sentence_vectors)

    # 4. 应用 TextRank 算法
    ranked_sentences = apply_textrank_ko(similarity_matrix, sentences)

    # 5. 选择排名最高的句子
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    # 6. 生成摘要
    summary = " ".join(summary_sentences)
    return summary

if __name__ == '__main__':
    # 示例用法
    article_text_en = calling_data("data\en\en1")
    summary_en = summarize_en(article_text_en, num_sentences=2)
    print("English Summary:", summary_en)

    # article_text_ko = """
    # 손흥민이 토트넘을 승리로 이끌었습니다. 
    # 그는 경기에서 두 골을 넣었습니다. 
    # 토트넘은 2대1로 아스날을 이겼습니다. 
    # 손흥민은 최고의 선수였습니다. 
    # 아스날은 좋은 경기를 펼치지 못했습니다.
    # """
    # summary_ko = summarize_ko(article_text_ko, num_sentences=2)
    # print("Korean Summary:", summary_ko)
