from typing import Dict, List, Iterable

def add_document(index: Dict[str, List[int]], doc_id: int, tokens: Iterable[str]) -> None:

    for term in tokens:
        postings = index.setdefault(term, [])

        if doc_id not in postings:
            postings.append(doc_id)
            postings.sort()

def remove_document(index: Dict[str, List[int]], doc_id: int) -> None:

    terms_to_delete = []
    for term, postings in index.items():
        if doc_id in postings:
            postings.remove(doc_id)
            #if only present in this doc
            if not postings:
                terms_to_delete.append(term)

    for term in terms_to_delete:
        del index[term]

