from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TFIDF_utils import sentence_segmentation, vectorize_tfidf, apply_textrank
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from arxivDataPreProcess import preprocess_text
import csv
import matplotlib.pyplot as plt

def read_abstract_from_pkl(path):
    """
    Reads abstract and title from pkl file.

    Args:
        path (str): pkl file path.

    Returns:
        pandas.DataFrame: DataFrame containing 'Abstract' and 'Title' columns.
    """
    df = pd.read_pickle(path)
    return df[['Abstract', 'Title']]

def summarize_en(text, num_sentences=5, reference_summary=None):
    """
    Extractive summarization of English articles.

    Args:
        text (str): English article text.
        num_sentences (int): Number of sentences in the summary.
        reference_summary (str): Reference summary for comparison.

    Returns:
        str: Core sentence of the article.
        metrics (dict): Evaluation metrics for the summary.
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
    Extracts key sentences from a Korean soccer game news article and generates a summary.

    Args:
        text (str): The text of the news article.
        num_sentences (int): The number of sentences to include in the summary.
    """

    # 1. Sentence segmentation
    sentences = sentence_segmentation(text)

    # 2. Sentence vectorization (TF-IDF)
    sentence_vectors = vectorize_tfidf(sentences)
    print(f'{sentence_vectors}')
    # 3. Construct similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. Apply TextRank algorithm
    ranked_sentences = apply_textrank(similarity_matrix, sentences)
    print(f'{ranked_sentences}')

    # 5. Select the highest-ranked sentences
    summary_sentences = sorted(ranked_sentences, key=lambda x: x[0])[:num_sentences]
    summary_sentences = [s[1] for s in summary_sentences]

    # 6. Generate summary
    summary = " ".join(summary_sentences)
    return summary

def find_similar_abstract(input_text, df, tfidf_matrix, tfidf_vectorizer):
    """
    Finds the most similar abstract in the dataframe to the input text using a pre-computed TF-IDF matrix.

    Args:
        input_text (str): The input text.
        df (pd.DataFrame): The dataframe containing abstracts and titles.
        tfidf_matrix (sparse matrix): Pre-computed TF-IDF matrix for the abstracts.
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.

    Returns:
        tuple: The most similar abstract and its title.
    """
    input_vector = tfidf_vectorizer.transform([input_text])

    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)

    most_similar_index = cosine_similarities.argmax()

    abstract = df['Abstract'][most_similar_index]
    title = df['Title'][most_similar_index]
    summary, metrics = summarize_en(abstract, num_sentences=2)

    return summary, title, metrics

def get_tfidf_vectors(input_text, df):
    """
    Returns TF-IDF matrix, query vector, and the most similar abstract text.
    """
    processed_abstracts = df['Abstract'].apply(preprocess_text)
    joined_abstracts = [' '.join(tokens) for tokens in processed_abstracts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(joined_abstracts)

    processed_query = ' '.join(preprocess_text(input_text))
    query_vector = vectorizer.transform([processed_query])

    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    best_idx = cosine_similarities.argmax()
    best_abstract = df['Abstract'].iloc[best_idx]

    return tfidf_matrix, query_vector, best_abstract

if __name__ == '__main__':
    df = read_abstract_from_pkl("data/arxiv_papers.pkl")

    # Create TF-IDF vectorizer and matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Abstract'])

    keywords = [
        "artificial intelligence",
        "machine learning",
        "natural language processing",
        "deep learning",
        "computer vision",
        "data mining",
        "information retrieval",
        "knowledge representation",
        "reasoning",
        "planning",
        "robotics",
        "cognitive science",
        "neural networks",
        "fuzzy logic",
        "expert systems",
        "big data",
        "cloud computing",
        "internet of things",
        "cybersecurity",
        "bioinformatics"
    ]

    all_metrics = []
    for keyword in keywords:
        most_similar_abstract, title, metrics = find_similar_abstract(keyword, df, tfidf_matrix, tfidf_vectorizer)
        all_metrics.append(metrics)

        print(f"Keyword: {keyword}")
        print("Most similar abstract:", most_similar_abstract)
        print("Title:", title)
        print("Metrics:", metrics)
        print("-" * 20)

    # Calculate average metrics
    avg_metrics = {}
    if all_metrics:  # Check if all_metrics is not empty
        for metric in all_metrics[0].keys():
            if metric != 'reference_similarity':
                avg_metrics[metric] = sum([m[metric] for m in all_metrics if m[metric] is not None]) / len(all_metrics)

        print("Average Metrics:", avg_metrics)

        # Visualize metric fluctuations (example for cosine_similarity)
        cosine_similarities = [m['cosine_similarity'] for m in all_metrics]
        plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
        plt.plot(keywords, cosine_similarities)
        plt.xlabel("Keywords")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity Fluctuation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No keywords found.")
