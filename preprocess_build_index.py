import re
from collections import defaultdict
from nltk.corpus import stopwords
import spacy
import nltk
import unicodedata
import json


corpus = [
    "Information retrieval is about finding relevant documents.",
    "An inverted index maps terms to documents.",
    "Text preprocessing includes tokenization and normalization.",
    "Python is a great language for text processing.",
    "Indexation is a key concept in search engines.",
    "Stopwords are common words that are often removed.",
    "Document similarity can be computed using vectors.",
    "The corpus we use is very small but illustrative.",
    "Tokenization splits text into individual words.",
    "Normalization may include lowercasing and removing punctuation.",
    "Stemming and lemmatization reduce words to their base form.",
    "Query expansion can improve search results.",
    "Term frequency is important in ranking models.",
    "Inverse document frequency helps downweight common terms.",
    "Vector space models represent documents as points in space.",
    "Word embeddings capture semantic relationships.",
    "Evaluation of IR systems uses metrics like precision and recall.",
    "Relevance feedback allows users to refine their queries.",
    "Language models are increasingly used in search.",
    "This is the last document in our tiny corpus."
]



nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

def preprocess(text: str):

    #unify case
    text = text.lower()

    #remove accents
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    #remove punctuation
    text = re.sub(r"[^a-z0-9]", " ", text)

    #tokenisation
    tokens = text.split()

    #remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    #lemmatisation
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]

    return lemmas


preprocessed_corpus = [preprocess(doc) for doc in corpus]

def get_preprocessed_corpus():
    return preprocessed_corpus

def build_inverted_index(preprocessed_docs):
    
    inverted_index = defaultdict(set)  

    for doc_id, tokens in enumerate(preprocessed_docs):
        for term in tokens:
            inverted_index[term].add(doc_id)

    inverted_index = {
        term: sorted(list(doc_ids))
        for term, doc_ids in inverted_index.items()
    }
    return inverted_index


inverted_index = build_inverted_index(preprocessed_corpus)


def print_index_sample(index, max_terms=20):

    print("=== Inverted Index (sample) ===")
    for i, (term, doc_ids) in enumerate(index.items()):
        if i >= max_terms:
            break
        print(f"{term:20s} -> {doc_ids}")

def save_inverted_index_json(index, filename="inverted_index.json"):

    sorted_index = dict(sorted(index.items(), key=lambda x: x[0]))
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sorted_index, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":

    print("=== Preprocessed corpus ===")
    for doc_id, tokens in enumerate(preprocessed_corpus):
        print(f"Doc {doc_id:02d}: {tokens}")
    print()

    print_index_sample(inverted_index, max_terms=25)
    save_inverted_index_json(inverted_index)

