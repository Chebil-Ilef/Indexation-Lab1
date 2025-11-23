import time
from preprocess_build_index import get_preprocessed_corpus, build_inverted_index
from parallel_indexing import build_inverted_index_parallel
from compression import compress_index, decompress_index
from maintenance import add_document, remove_document
from metrics import deep_size, format_bytes

def main():
    preprocessed_corpus = get_preprocessed_corpus()

    # sequential indexing
    t0 = time.perf_counter()
    index_seq = build_inverted_index(preprocessed_corpus)
    t1 = time.perf_counter()
    seq_time = t1 - t0
    print(f"[Sequential] Index built in {seq_time:.6f} seconds")

    # parallel indexing
    t0 = time.perf_counter()
    index_par = build_inverted_index_parallel(preprocessed_corpus)
    t1 = time.perf_counter()
    par_time = t1 - t0
    print(f"[Parallel]   Index built in {par_time:.6f} seconds")

    # sanity check: same result?
    print(f"Sequential == Parallel index ? {index_seq == index_par}")

    # compression vs size
    size_before = deep_size(index_seq)
    compressed_index = compress_index(index_seq)
    size_after = deep_size(compressed_index)
    
    actual_before = sum(len(postings) * 4 for postings in index_seq.values())  # 4 bytes per int minimum
    actual_after = sum(len(byte_list) for byte_list in compressed_index.values())  # raw bytes

    print("\n[Memory]")
    print(f"Size before compression : {format_bytes(size_before)} (Python objects)")
    print(f"Size after  compression : {format_bytes(size_after)} (Python objects)")
    print(f"Actual data size before : {actual_before} bytes (uncompressed integers)")
    print(f"Actual data size after  : {actual_after} bytes (VByte compressed)")
    print(f"Compression ratio       : {actual_before / actual_after:.2f}x")

    # optional sanity check
    decompressed = decompress_index(compressed_index)
    print(f"Index recovered after decompression ? {decompressed == index_seq}")

    # maintenance example
    print("\n[Maintenance]")

    new_doc_id = len(preprocessed_corpus)
    new_doc_tokens = ["new", "document", "about", "information", "retrieval"]
    print(f"Adding doc {new_doc_id} with tokens: {new_doc_tokens}")
    add_document(index_seq, new_doc_id, new_doc_tokens)

    print("Removing doc 0 from index...")
    remove_document(index_seq, 0)

if __name__ == "__main__":
    main()
