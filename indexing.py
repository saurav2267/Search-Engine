from collections import defaultdict
from preprocessing import preprocess_text

def build_inverted_index(docs):
    """
    Given {doc_id: raw_text}, build an inverted index & doc length map.
    Returns:
        inverted_index: {term: {doc_id: tf}}
        doc_lengths:    {doc_id: total_tokens}
        collection_size: int (total # docs)
    """
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}

    for doc_id, text in docs.items():
        tokens = preprocess_text(text)
        doc_lengths[doc_id] = len(tokens)
        for term in tokens:
            inverted_index[term][doc_id] += 1

    return inverted_index, doc_lengths, len(docs)

def build_global_term_counts(inverted_index):
    """
    For Language Model:
    Returns a dict of {term: total_count_in_collection},
    plus the total number of tokens in the entire collection.
    """
    term_counts = {}
    total_tokens = 0

    for term, posting_list in inverted_index.items():
        term_sum = sum(posting_list.values())
        term_counts[term] = term_sum
        total_tokens += term_sum

    return term_counts, total_tokens 