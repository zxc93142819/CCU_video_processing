# Required Libraries
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import heapq
import os , cv2
import matplotlib.pyplot as plt

# Huffman Node Class
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
def build_huffman_tree(freqs):
    heap = [Node(sym, freq) for sym, freq in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

# Generate Huffman Codes
def generate_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_codes(node.left, prefix + "0", codebook)
        generate_codes(node.right, prefix + "1", codebook)
    return codebook

# Encode image using Huffman
def huffman_encode(data):
    freqs = Counter(data)
    tree = build_huffman_tree(freqs)
    codebook = generate_codes(tree)
    encoded = ''.join(codebook[val] for val in data)
    return encoded, codebook, freqs

# Decode image using Huffman codebook
def huffman_decode(encoded, codebook):
    rev_codebook = {v: k for k, v in codebook.items()}
    current_code = ""
    decoded = []
    for bit in encoded:
        current_code += bit
        if current_code in rev_codebook:
            decoded.append(rev_codebook[current_code])
            current_code = ""
    return np.array(decoded, dtype=np.uint8)

# Compression Ratio
def compression_ratio(original_bits, compressed_bits):
    return original_bits / len(compressed_bits)

# Image Utils
def load_image(path):
    return np.array(Image.open(path).convert('L'))

def save_image(data, path):
    Image.fromarray(data).save(path)

def flatten_image(img):
    return img.flatten().tolist()

# Main Test Routine for Two Images
def main(img_path):
    img = load_image(img_path)
    original_flat = flatten_image(img)

    # --- Huffman Coding Directly ---
    encoded, codebook, freqs = huffman_encode(original_flat)
    decoded_img = huffman_decode(encoded, codebook).reshape(img.shape)
    cr = compression_ratio(len(original_flat) * 8, encoded)

    print(img_path)
    print("Codebook:")
    for symbol in sorted(codebook):
        print(f"Symbol: {symbol}, Freq: {freqs[symbol] / len(original_flat)}, Code: {codebook[symbol]}")
    print(f"Compression Ratio: {cr:.4f}")
    print("------------------------------------------------------------------")
    cv2.imshow(img_path, img)
    cv2.imshow('decode of ' + img_path, decoded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main("test1.bmp")
main("test2.bmp")