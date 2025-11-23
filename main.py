# main.py
import time

from preprocessing import get_preprocessed_corpus
from indexing import build_inverted_index
from parallel_indexing import build_inverted_index_parallel
from compression import compress_index, decompress_index
from maintenance import add_document, remove_document
from metrics import deep_size, format_bytes

def main():
    preprocessed_corpus = get_preprocessed_corpus()

    # ===========================
    # 1) Sequential indexing
    # ===========================
    t0 = time.perf_counter()
    index_seq = build_inverted_index(preprocessed_corpus)
    t1 = time.perf_counter()
    seq_time = t1 - t0
    print(f"[Sequential] Index built in {seq_time:.6f} seconds")

    # ===========================
    # 2) Parallel indexing
    # ===========================
    t0 = time.perf_counter()
    index_par = build_inverted_index_parallel(preprocessed_corpus)
    t1 = time.perf_counter()
    par_time = t1 - t0
    print(f"[Parallel]   Index built in {par_time:.6f} seconds")

    # sanity check: same result?
    print(f"Sequential == Parallel index ? {index_seq == index_par}")

    # ===========================
    # 3) Compression vs size
    # ===========================
    size_before = deep_size(index_seq)
    compressed_index = compress_index(index_seq)
    size_after = deep_size(compressed_index)

    print("\n[Memory]")
    print(f"Size before compression : {format_bytes(size_before)}")
    print(f"Size after  compression : {format_bytes(size_after)}")

    # optional sanity check
    decompressed = decompress_index(compressed_index)
    print(f"Index recovered after decompression ? {decompressed == index_seq}")

    # ===========================
    # 4) Maintenance example
    # ===========================
    print("\n[Maintenance]")

    new_doc_id = len(preprocessed_corpus)
    new_doc_tokens = ["new", "document", "about", "information", "retrieval"]
    print(f"Adding doc {new_doc_id} with tokens: {new_doc_tokens}")
    add_document(index_seq, new_doc_id, new_doc_tokens)

    print("Removing doc 0 from index...")
    remove_document(index_seq, 0)

    # ===========================
    # 5) Simple discussion output
    # ===========================
    print("\n[Discussion]")
    print(f"- Parallelization speedup (seq/par): "
          f"{seq_time:.6f} / {par_time:.6f} (ratio ~ {seq_time / par_time if par_time > 0 else 'N/A'})")
    print("- Compression reduces memory usage but requires extra CPU time "
          "for compression/decompression (space/time tradeoff).")
    print("- Parallel indexing improves indexing time but adds overhead for "
          "process creation and merging partial indexes.")

if __name__ == "__main__":
    main()
