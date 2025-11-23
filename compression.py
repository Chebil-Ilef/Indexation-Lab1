from typing import Dict, List


def gap_encode(postings: List[int]) -> List[int]:
    if not postings:
        return []
    gaps = [postings[0]]
    for i in range(1, len(postings)):
        gaps.append(postings[i] - postings[i - 1])
    return gaps


def gap_decode(gaps: List[int]) -> List[int]:

    if not gaps:
        return []
    postings = [gaps[0]]
    for g in gaps[1:]:
        postings.append(postings[-1] + g)
    return postings


def vbyte_encode_number(n: int) -> List[int]:

    if n < 0:
        raise ValueError("VByte only supports non-negative integers")

    bytes_ = []
    
    # Encode in little-endian: least significant bits first
    while n >= 128:
        bytes_.append(n & 0x7F)  # Take 7 bits, keep MSB=0 (continuation)
        n >>= 7
    
    # Last byte: set MSB=1 to mark end
    bytes_.append(n | 0x80)
    
    # returns a list of bytes (ints in [0, 255])
    return bytes_


def vbyte_decode_stream(byte_list: List[int]) -> List[int]:

    numbers: List[int] = []
    current = 0
    shift = 0

    for b in byte_list:
        # Extract 7 bits and place them at the correct position (little-endian)
        current |= (b & 0x7F) << shift
        
        # if MSB is 1, this is the last byte of the number
        if b & 0x80:
            numbers.append(current)
            current = 0
            shift = 0
        else:
            shift += 7

    if shift != 0:
        raise ValueError("Incomplete VByte stream")

    return numbers


def vbyte_encode_postings(gaps: List[int]) -> List[int]:

    result: List[int] = []
    for g in gaps:
        result.extend(vbyte_encode_number(g))
    return result


def vbyte_decode_postings(byte_list: List[int]) -> List[int]:

    return vbyte_decode_stream(byte_list)


def compress_index(index: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """
    Compress each postings list with GAP + VByte.

    index: term -> sorted list of docIDs
    returns: term -> list of bytes (ints)
    """
    compressed: Dict[str, List[int]] = {}
    for term, postings in index.items():
        gaps = gap_encode(postings)
        compressed_bytes = vbyte_encode_postings(gaps)
        compressed[term] = compressed_bytes
    return compressed


def decompress_index(compressed_index: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """
    Decompress an index compressed with GAP + VByte back to:
    term -> sorted list of docIDs
    """
    decompressed: Dict[str, List[int]] = {}
    for term, byte_list in compressed_index.items():
        gaps = vbyte_decode_postings(byte_list)
        postings = gap_decode(gaps)
        decompressed[term] = postings
    return decompressed
