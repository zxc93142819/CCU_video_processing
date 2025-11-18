import cv2
import numpy as np
import os, glob
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_image_sequence(folder, grayscale=True):
    frame_count = frame_count = len(glob.glob(os.path.join(folder, "*.bmp")))
    images = []
    for i in range(frame_count):
        image = cv2.imread(os.path.join(folder, f"{i}.bmp"))
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    return images

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_sequence(images, folder):
    ensure_dir(folder)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(folder, f"{i}.bmp"), img)

def bilinear_upscale(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

# === Motion Estimation Algorithms ===
def full_search(reference_frame1, reference_frame2, current_frame, search_range=4):
    reference_frame1 = reference_frame1.astype(np.float64)
    reference_frame2 = reference_frame2.astype(np.float64)
    m, n, = reference_frame1.shape
    block_size = 16
    block_number = (m * n) // (block_size * block_size)

    padding_reference_frame1 = cv2.copyMakeBorder(reference_frame1, search_range, search_range, search_range,
                                                  search_range, cv2.BORDER_REPLICATE)

    padding_reference_frame2 = cv2.copyMakeBorder(reference_frame2, search_range, search_range, search_range,
                                                  search_range, cv2.BORDER_REPLICATE)

    current_frame = current_frame.astype(np.float64)
    final_image = current_frame.copy()

    for i in range(m // block_size):
        for j in range(n // block_size):
            min_mad = float('inf')
            for p in range(-search_range, search_range + 1):
                for q in range(-search_range, search_range + 1):
                    current_block = current_frame[i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size]
                    reference_block = padding_reference_frame1[
                                      search_range + i * block_size + p:(search_range + i + 1) * block_size + p,
                                      search_range + j * block_size + q:(search_range + j + 1) * block_size + q]
                    mad1 = np.sum(
                        np.abs(current_block - reference_block[:current_block.shape[0], :current_block.shape[1]]))

                    reference_block = padding_reference_frame2[
                                      search_range + i * block_size + p:(search_range + i + 1) * block_size + p,
                                      search_range + j * block_size + q:(search_range + j + 1) * block_size + q]

                    mad2 = np.sum(
                        np.abs(current_block - reference_block[:current_block.shape[0], :current_block.shape[1]]))

                    if p == -search_range and q == -search_range:
                        if mad1 < mad2:
                            min_mad = mad1
                            final_image[(i * block_size):(i * block_size + block_size),
                            (j * block_size):(j * block_size + block_size)] = padding_reference_frame1[
                                                                              (i * block_size + search_range + p):(
                                                                                      i * block_size + block_size + search_range + p),
                                                                              (j * block_size + search_range + q):(
                                                                                      j * block_size + block_size + search_range + q)]
                        else:
                            min_mad = mad2
                            final_image[(i * block_size):(i * block_size + block_size),
                            (j * block_size):(j * block_size + block_size)] = padding_reference_frame2[
                                                                              (i * block_size + search_range + p):(
                                                                                      i * block_size + block_size + search_range + p),
                                                                              (j * block_size + search_range + q):(
                                                                                      j * block_size + block_size + search_range + q)]
                        continue

                    if min_mad > mad1:
                        min_mad = mad1
                        final_image[(i * block_size):(i * block_size + block_size),
                        (j * block_size):(j * block_size + block_size)] = padding_reference_frame1[
                                                                          (i * block_size + search_range + p):(
                                                                                  i * block_size + block_size + search_range + p),
                                                                          (j * block_size + search_range + q):(
                                                                                  j * block_size + block_size + search_range + q)]

                    if min_mad > mad2:
                        min_mad = mad2
                        final_image[(i * block_size):(i * block_size + block_size),
                        (j * block_size):(j * block_size + block_size)] = padding_reference_frame2[
                                                                          (i * block_size + search_range + p):(
                                                                                  i * block_size + block_size + search_range + p),
                                                                          (j * block_size + search_range + q):(
                                                                                  j * block_size + block_size + search_range + q)]

    final_image = final_image.astype(np.uint8)
    return final_image

# === Reconstruction Methods ===
def interpolate_only(distorted_seq, target_size):
    return [bilinear_upscale(img, target_size) if img.shape != target_size else img for img in distorted_seq]

def interpolate_with_motion(distorted_seq, key_indices, method):
    output_seq = distorted_seq.copy()
    for i in range(len(distorted_seq)):
        if i not in key_indices:
            prev_key = max([k for k in key_indices if k < i], default=None)
            next_key = min([k for k in key_indices if k > i], default=None)
            if prev_key is not None and next_key is not None:
                # print(i , prev_key, next_key)
                output_seq[i] = method(output_seq[prev_key], output_seq[next_key], output_seq[i])
    return output_seq

def run_reconstruction(sequence = "s1", key_interval=4):
    original_dir = f"./Hw4_test sequences/{sequence}/original"
    distorted_dir = f"./Hw4_test sequences/{sequence}/distortion"
    output_root = f"./outputs_{sequence}"
    original = load_image_sequence(original_dir)
    distorted = load_image_sequence(distorted_dir)
    frame_count = len(original)
    target_size = (original[0].shape[1], original[0].shape[0])

    key_indices = [i for i in range(frame_count) if original[i].shape == distorted[i].shape]
    distorted_upscaled = interpolate_only(distorted, target_size)

    # Method 1: Bilinear only
    save_image_sequence(distorted_upscaled, os.path.join(output_root, "method1_bilinear"))

    # Method 2: Bilinear + Full Search
    full_output = interpolate_with_motion(distorted_upscaled, key_indices, full_search)
    save_image_sequence(full_output, os.path.join(output_root, "method2_full_search"))

    # Evaluate PSNR
    print(f"=== sequence: {sequence} ===")
    print("PSNR values for each method:")
    for i in range(frame_count):
        print(f"frame {i}:" , "Only Bilinear interpolation:" , psnr(original[i], distorted_upscaled[i]) , 
              ", Full Search:" , psnr(original[i], full_output[i]))
    print()

run_reconstruction("s1")
run_reconstruction("s2")