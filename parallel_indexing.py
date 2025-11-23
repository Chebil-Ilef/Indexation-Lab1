from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

DocPair = Tuple[int, List[str]]
IndexType = Dict[str, Set[int]]

def _build_partial_index(doc_pairs: List[DocPair]) -> IndexType:

    index: IndexType = defaultdict(set)
    for doc_id, tokens in doc_pairs:
        for term in tokens:
            index[term].add(doc_id)
    return index

def _merge_indexes(base: IndexType, other: IndexType) -> None:
  
    for term, doc_ids in other.items():
        base[term].update(doc_ids)

def build_inverted_index_parallel(preprocessed_docs: List[List[str]], n_workers: int | None = None) -> Dict[str, List[int]]:

    n_workers= 4

    print(f"Building inverted index using {n_workers} workers...")

    doc_pairs: List[DocPair] = list(enumerate(preprocessed_docs))
    chunk_size = max(1, len(doc_pairs) // n_workers)

    print(f"Chunk size: {chunk_size}")

    chunks: List[List[DocPair]] = [
        doc_pairs[i:i + chunk_size]
        for i in range(0, len(doc_pairs), chunk_size)
    ]

    print(f"Number of chunks: {len(chunks)}")

    with Pool(processes=n_workers) as pool:
        partial_indexes: List[IndexType] = pool.map(_build_partial_index, chunks)

    global_index: IndexType = defaultdict(set)
    for part in partial_indexes:
        _merge_indexes(global_index, part)

    return {term: sorted(doc_ids) for term, doc_ids in global_index.items()}
