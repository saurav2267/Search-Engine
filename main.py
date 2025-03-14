import os
from parsers import parse_cranfield_documents, parse_cranfield_queries
from indexing import build_inverted_index, build_global_term_counts
from ranking_models import rank_vsm, rank_bm25, rank_language_model
from output import generate_trec_run

def main():
    # Ensure Outputs directory exists
    os.makedirs("Outputs", exist_ok=True)

    # 1. Parse documents and queries
    docs = parse_cranfield_documents("cranfield-trec-dataset-main/cran.all.1400.xml")
    queries = parse_cranfield_queries("cranfield-trec-dataset-main/cran.qry.xml")
    print(f"Parsed {len(docs)} documents and {len(queries)} queries.")

    # 2. Build index
    inverted_index, doc_lengths, N = build_inverted_index(docs)
    print(f"Built inverted index with {len(inverted_index)} unique terms.")

    # 3. Precompute df for each term
    df = {term: len(postings) for term, postings in inverted_index.items()}

    # 4. For Language Model, build global term counts
    term_counts, total_tokens = build_global_term_counts(inverted_index)

    # 5. Wrappers with optimized parameters for each model
    def vsm_wrapper(q_text):
        return rank_vsm(q_text, inverted_index, df, N, doc_lengths)

    def bm25_wrapper(q_text):
        return rank_bm25(q_text, inverted_index, df, N, doc_lengths, k1=1.5, b=0.75)

    def lm_wrapper(q_text):
        return rank_language_model(q_text, inverted_index, df, doc_lengths,
                                 term_counts, total_tokens, mu=2500)

    # 6. Generate TREC run files in Outputs directory
    generate_trec_run(queries, vsm_wrapper,  "VSM_Run", "Outputs/vsm_run.txt",  top_k=100)
    generate_trec_run(queries, bm25_wrapper, "BM25_Run", "Outputs/bm25_run.txt", top_k=100)
    generate_trec_run(queries, lm_wrapper,   "LM_Run",   "Outputs/lm_run.txt",   top_k=100)

    print("\nRun files generated in Outputs directory:")
    print("  - Outputs/vsm_run.txt")
    print("  - Outputs/bm25_run.txt")
    print("  - Outputs/lm_run.txt")
    print("\nTo evaluate with trec_eval in WSL:")
    print("1. Open WSL terminal")
    print("2. Navigate to your project directory")
    print("3. Run: ./trec_eval-linux-amd64 -m map -m P.5 -m ndcg cranqrel.trec.txt vsm_run.txt")
    print("4. Repeat for bm25_run.txt and lm_run.txt")

if __name__ == "__main__":
    main()
