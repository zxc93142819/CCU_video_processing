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
    return np.array(decoded, dtype=np.uint32)

def add(a, b):
    p = int(a) + int(b)
    if p > 255:
        p = 255
    return p

def sub(a, b):
    p = int(a) - int(b)
    if p < 0:
        p = 0
    return p

def pred(img, row, col, x):
    pred_img = np.zeros((row, col), dtype = 'uint8')
    for i in range(0, row):
        for j in range(0, col):
            A = 0
            B = 0
            C = 0
            if i != 0 and j != 0:
                A = img[i][j - 1]
                B = img[i-1][j]
                C = img[i-1][j-1]
            if x == 1:
                P = A
            elif x == 2:
                P = B
            elif x == 3:
                P = C
            elif x == 4:
                #P = A + B - C
                f1 = add(A, B)
                P = sub(f1, C)
            elif x == 5:
                #P = A + (B - C) / 2
                f1 = sub(B, C)
                f2 = f1 / 2
                P = add(A, f2)
            elif x == 6:
                #P = B + (A - C) / 2
                f1 = sub(A, C)
                f2 = f1 / 2
                P = add(B, f2)
            elif x == 7:
                #P = (A + B) / 2
                f1 = add(A, B)
                P = f1 / 2
            else:
                P = 0
            pred_img[i][j] = sub(img[i][j], P)
    return pred_img

# Prediction Modes
def predict(image, mode):
    pred = np.zeros_like(image, dtype=int)
    height, width = image.shape

    for y in range(0, height):
        for x in range(0, width):
            A = 0
            B = 0
            C = 0
            if(y >= 1 and x >= 1) :
                A = int(image[y, x - 1])
                B = int(image[y - 1, x])
                C = int(image[y - 1, x - 1])

            if mode == 0:
                p = 0
            elif mode == 1:
                p = A
            elif mode == 2:
                p = B
            elif mode == 3:
                p = C
            elif mode == 4:
                #P = A + B - C
                f1 = add(A, B)
                p = sub(f1, C)
            elif mode == 5:
                #P = A + (B - C) / 2
                f1 = sub(B, C)
                f2 = f1 / 2
                p = add(A, f2)
            elif mode == 6:
                #P = B + (A - C) / 2
                f1 = sub(A, C)
                f2 = f1 / 2
                p = add(B, f2)
            elif mode == 7:
                #P = (A + B) / 2
                f1 = add(A, B)
                p = f1 / 2
            else:
                p = 0
            pred[y, x] = sub(image[y, x] , p)
    return pred

# Restore image from prediction residuals
def restore(predicted, original, mode):
    restored = np.copy(original)
    height, width = original.shape

    for y in range(0, height):
        for x in range(0, width):
            A = 0
            B = 0
            C = 0
            if(y >= 1 and x >= 1) :
                A = original[y][x - 1]
                B = original[y - 1][x]
                C = original[y - 1][x - 1]
            if mode == 1:
                p = A
            elif mode == 2:
                p = B
            elif mode == 3:
                p = C
            elif mode == 4:
                # P = A + B - C
                f1 = add(A, B)
                p = sub(f1, C)
            elif mode == 5:
                # P = A + (B - C) / 2
                f1 = sub(B, C)
                f2 = f1 / 2
                p = add(A, f2)
            elif mode == 6:
                # P = B + (A - C) / 2
                f1 = sub(A, C)
                f2 = f1 / 2
                p = add(B, f2)
            elif mode == 7:
                # P = (A + B) / 2
                f1 = add(A, B)
                p = f1 / 2
            else:
                p = 0
            restored[y, x] = add(predicted[y, x], p)
    return restored.astype(np.uint8)

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
    output = []

    print(img_path)
    for mode in range(8):
        pred = predict(img, mode)
        # print(pred)
        pred_flat = pred.flatten().tolist()
        # shifted = [val + 255 for val in pred_flat]  # Ensure values are non-negative
        encoded, codebook, freqs = huffman_encode(pred_flat)
        # encoded, codebook, freqs = huffman_encode(shifted)
        cr = compression_ratio(len(original_flat) * 8, encoded)

        print("Codebook:")
        print(f"Predicted Huffman Codebook (mode: {mode}): ")
        for symbol in sorted(codebook):
            print(f"Symbol: {symbol}, Freq: {freqs[symbol] / len(original_flat)}, Code: {codebook[symbol]}")
        print(f"Compression Ratio (mode: {mode}): {cr:.4f}")

        # Decode and restore image
        decoded = huffman_decode(encoded, codebook)
        # decoded = decoded - 255  # Shift back
        restored_pred = decoded.reshape(img.shape)
        restored_img = restore(restored_pred, img, mode)
        print("------------------------------------------------------------------")    
        output.append(restored_img)
    
    cv2.imshow(img_path, img)
    for i in range(len(output)) :
        cv2.imshow(f'decode of {img_path} (mode {i})' , output[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main("test1.bmp")
main("test2.bmp")