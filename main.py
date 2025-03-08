#!/usr/bin/env python3
"""
CA6005 / CSC1121 - Assignment 1
Simple IR System for the Cranfield Collection
========================================================

This script:
    1. Parses the Cranfield documents (cran.all.1400.xml) and queries (cran.qry.xml)
    2. Pre-processes text (tokenize, lowercase, remove basic stopwords, etc.)
    3. Builds an inverted index
    4. Implements:
        - VSM (TF–IDF-like) ranking
        - BM25 ranking
        - Language Model ranking (Dirichlet smoothing)
    5. Generates TREC-style output run files for each model

Usage:
    python assignment1.py

Then run `trec_eval` on the output files:
    trec_eval <PATH_TO>/cranqrel.trec.txt Outputs/vsm_run.txt
    trec_eval <PATH_TO>/cranqrel.trec.txt Outputs/bm25_run.txt
    trec_eval <PATH_TO>/cranqrel.trec.txt Outputs/lm_run.txt
"""

import os
from parsers import parse_cranfield_documents, parse_cranfield_queries
from indexing import build_inverted_index, build_global_term_counts
from ranking_models import rank_vsm, rank_bm25, rank_language_model
from output import generate_trec_run

def main():
    # Ensure Outputs directory exists
    os.makedirs("Outputs", exist_ok=True)

    # 1. Parse
    docs = parse_cranfield_documents("C:/Users/sahil/Search-Engine/cranfield-trec-dataset-main/cran.all.1400.xml")
    queries = parse_cranfield_queries("C:/Users/sahil/Search-Engine/cranfield-trec-dataset-main/cran.qry.xml")
    print(f"Parsed {len(docs)} documents and {len(queries)} queries.")

    # 2. Build index
    inverted_index, doc_lengths, N = build_inverted_index(docs)
    print(f"Built inverted index with {len(inverted_index)} unique terms.")

    # 3. Precompute df for each term
    df = {term: len(postings) for term, postings in inverted_index.items()}

    # 4. For Language Model, build global term counts
    term_counts, total_tokens = build_global_term_counts(inverted_index)

    # 5. Wrappers so we can pass only (query_text) to rank functions
    def vsm_wrapper(q_text):
        return rank_vsm(q_text, inverted_index, df, N, doc_lengths)

    def bm25_wrapper(q_text):
        return rank_bm25(q_text, inverted_index, df, N, doc_lengths, k1=1.2, b=0.75)

    def lm_wrapper(q_text):
        return rank_language_model(q_text, inverted_index, df, doc_lengths,
                                 term_counts, total_tokens, mu=2000)

    # 6. Generate TREC run files in Outputs directory
    generate_trec_run(queries, vsm_wrapper,  "VSM_Run", "Outputs/vsm_run.txt",  top_k=100)
    generate_trec_run(queries, bm25_wrapper, "BM25_Run", "Outputs/bm25_run.txt", top_k=100)
    generate_trec_run(queries, lm_wrapper,   "LM_Run",   "Outputs/lm_run.txt",   top_k=100)

    print("Run files generated in Outputs directory:")
    print("  - Outputs/vsm_run.txt")
    print("  - Outputs/bm25_run.txt")
    print("  - Outputs/lm_run.txt")
    print("\nNow evaluate with trec_eval, for example:")
    print("  trec_eval ../data/cranqrel.trec.txt Outputs/vsm_run.txt")

if __name__ == "__main__":
    main()
