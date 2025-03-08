def generate_trec_run(queries, rank_function, run_id, output_file_path, top_k=100):
    """
    Produces a file of lines in TREC format:
       query_id  iter  doc_id  rank  score  run_id
    """
    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for q_id, q_text in queries.items():
            ranked_docs = rank_function(q_text)
            for rank_idx, (doc_id, score) in enumerate(ranked_docs[:top_k], start=1):
                # TREC format: q_id iter doc_id rank score run_id
                line = f"{q_id} 0 {doc_id} {rank_idx} {score} {run_id}\n"
                fout.write(line) 