import math
from collections import Counter
from preprocessing import preprocess_text

def rank_vsm(query_text, inverted_index, df, N, doc_lengths):
    """
    TF-IDF (log-based IDF) scoring, partial or no normalization for simplicity.
    Returns list of (doc_id, score) sorted descending by score.
    """
    query_tokens = preprocess_text(query_text)
    query_tf = Counter(query_tokens)
    scores = {}

    for term, qf in query_tf.items():
        if term not in inverted_index:
            continue

        # IDF (using log base 2 for demonstration)
        idf = math.log2((N / (df[term] + 1)))
        w_qt = qf * idf

        # Accumulate scores for docs that have this term
        for doc_id, tf_dt in inverted_index[term].items():
            w_dt = tf_dt * idf
            scores[doc_id] = scores.get(doc_id, 0) + (w_dt * w_qt)  # dot-product component

    # Sort by descending score
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def rank_bm25(query_text, inverted_index, df, N, doc_lengths, k1=1.2, b=0.75):
    """
    BM25 ranking.
    Returns list of (doc_id, score) sorted descending by BM25 score.
    """
    query_tokens = preprocess_text(query_text)
    query_tf = Counter(query_tokens)

    avgdl = sum(doc_lengths.values()) / float(N)
    scores = {}

    for term, qf in query_tf.items():
        if term not in inverted_index:
            continue

        df_t = df[term]
        # IDF part
        idf = math.log2((N - df_t + 0.5) / (df_t + 0.5) + 1)

        posting_list = inverted_index[term]
        for doc_id, f_dt in posting_list.items():
            doc_len = doc_lengths[doc_id]
            numerator = f_dt * (k1 + 1)
            denominator = f_dt + k1 * (1 - b + b * (doc_len / avgdl))
            score = idf * (numerator / denominator)
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def rank_language_model(query_text, inverted_index, df, doc_lengths,
                       term_counts, total_tokens, mu=2000):
    """
    Dirichlet smoothing.
    For each document, sum log( P(t|d) ) for t in query.
    Returns list of (doc_id, score) in descending order (score = log-prob).
    """
    query_tokens = preprocess_text(query_text)
    scores = {}

    for doc_id, doc_len in doc_lengths.items():
        log_prob_sum = 0.0

        for t in query_tokens:
            f_dt = 0
            if t in inverted_index and doc_id in inverted_index[t]:
                f_dt = inverted_index[t][doc_id]
            c_t = term_counts.get(t, 0)  # total freq of term t in entire collection

            # Dirichlet smoothing formula
            p_t_d = (f_dt + mu * (c_t / total_tokens)) / (doc_len + mu)
            if p_t_d > 0:
                log_prob_sum += math.log(p_t_d)

        scores[doc_id] = log_prob_sum

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs 