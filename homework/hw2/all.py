import numpy as np
import cv2
import os
import time

def full_search(reference_frame, current_frame, search_range):
    m, n = reference_frame.shape
    block_size = 16

    padding_reference_frame = cv2.copyMakeBorder(reference_frame, search_range, search_range, search_range,
                                                 search_range, cv2.BORDER_REPLICATE)

    vector = np.zeros((m // block_size, n // block_size, 2))
    padding_reference_frame = padding_reference_frame.astype(np.float64)
    current_frame = current_frame.astype(np.float64)

    for i in range(m // block_size):
        for j in range(n // block_size):
            min_mad = float('inf')
            for p in range(-search_range, search_range + 1):
                for q in range(-search_range, search_range + 1):
                    current_block = current_frame[i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size]
                    reference_block = padding_reference_frame[
                                      search_range + i * block_size + p:(search_range + i + 1) * block_size + p,
                                      search_range + j * block_size + q:(search_range + j + 1) * block_size + q]
                    mad = np.sum(
                        np.abs(current_block - reference_block[:current_block.shape[0], :current_block.shape[1]]))

                    if mad < min_mad:
                        min_mad = mad
                        vector[i, j, 0] = p
                        vector[i, j, 1] = q

    motion = vector
    return motion


frame_number = [91]
seq1 = []

for i in range(frame_number[0]):
    filename = f"./s1/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq1.append(image)

seq1 = np.stack(seq1, axis=-1)

frame_number2 = [30]
seq2 = []

for i in range(1, frame_number2[0]+1):
    if i < 10:
        filename = f"./s2/0{i}.bmp"
    else:
        filename = f"./s2/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq2.append(image)

seq2 = np.stack(seq2, axis=-1)
start = time.time()

# Full search
print("Do Full Search")

# seq1 search range1
search_range = [8]  # Change this value according to your search_range
block_size = 8

search_range1_seq1_reconstruct_full_search = np.copy(seq1)
search_range1_seq1_psnr_full_search = np.zeros(frame_number[0])

for f in range(frame_number[0] - 1):
    motion = full_search(search_range1_seq1_reconstruct_full_search[..., f], seq1[..., f + 1], search_range[0])
    block_x_num, block_y_num, x_y = motion.shape
    padding_reference_frame = cv2.copyMakeBorder(search_range1_seq1_reconstruct_full_search[..., f], search_range[0], search_range[0], search_range[0], search_range[0], cv2.BORDER_REPLICATE)
    padding_reference_frame = padding_reference_frame.astype(np.float64)

    for i in range(block_x_num):
        for j in range(block_y_num):
            search_range1_seq1_reconstruct_full_search[(i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size), f + 1] = 
            padding_reference_frame[search_range[0] + (i * block_size) + int(motion[i, j, 0]):search_range[0] + ((i + 1) * block_size) + 
                                    int(motion[i, j, 0]), search_range[0] + (j * block_size) + int(motion[i, j, 1]):search_range[0] + ((j + 1) * block_size) + int(motion[i, j, 1])]

for i in range(frame_number[0] - 1):
    search_range1_seq1_psnr_full_search[i] = cv2.PSNR(search_range1_seq1_reconstruct_full_search[..., i+1], seq1[..., i+1])

output_dir = 's1_full_search_range_8_size_8_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, frame_number[0]):
    output_filename = os.path.join(output_dir, f"reconstructed_frame_{i}.bmp")
    cv2.imwrite(output_filename, search_range1_seq1_reconstruct_full_search[..., i])

with open("s1_full_search_range_8_size_8_output/psnr_list.txt", 'w') as text_file:
    for i in range(0, frame_number[0] - 1):
        text_file.write(f"Reconstructed frame {i+1} psnr: {search_range1_seq1_psnr_full_search[i]}\n")

def two_d_logarithm_search(reference_frame, current_frame, search_range):
    m, n = reference_frame.shape
    block_size = 16

    padding_reference_frame = cv2.copyMakeBorder(reference_frame, search_range, search_range, search_range,
                                                 search_range, cv2.BORDER_REPLICATE)

    vector = np.zeros((m // block_size, n // block_size, 2))
    padding_reference_frame = padding_reference_frame.astype(np.float64)
    current_frame = current_frame.astype(np.float64)

    for i in range(m // block_size):
        for j in range(n // block_size):
            x = i * block_size
            y = j * block_size
            move_x = 0
            move_y = 0
            min_mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] - padding_reference_frame[
                                                                                        x + search_range:x + block_size + search_range,
                                                                                        y + search_range:y + block_size + search_range]))
            while True:
                tmp_move_x = 0
                tmp_move_y = 0

                if x + move_x + search_range + 2 < x + 2 * search_range:
                    mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] - padding_reference_frame[
                                                                                            x + search_range + move_x + 2:x + block_size + search_range + move_x + 2,
                                                                                            y + search_range + move_y:y + block_size + search_range + move_y]))
                    if mad < min_mad:
                        tmp_move_x = 2
                        min_mad = mad

                if x + move_x + search_range - 2 >= x:
                    mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] - padding_reference_frame[
                                                                                            x + search_range + move_x - 2:x + block_size + search_range + move_x - 2,
                                                                                            y + search_range + move_y:y + block_size + search_range + move_y]))
                    if mad < min_mad:
                        tmp_move_x = -2
                        min_mad = mad

                if y + move_y + search_range + 2 < y + 2 * search_range:
                    mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] - padding_reference_frame[
                                                                                            x + search_range + move_x:x + block_size + search_range + move_x,
                                                                                            y + search_range + move_y + 2:y + block_size + search_range + move_y + 2]))
                    if mad < min_mad:
                        tmp_move_y = 2
                        min_mad = mad

                if y + move_y + search_range - 2 >= y:
                    mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] - padding_reference_frame[
                                                                                            x + search_range + move_x:x + block_size + search_range + move_x,
                                                                                            y + search_range + move_y - 2:y + block_size + search_range + move_y - 2]))
                    if mad < min_mad:
                        tmp_move_y = -2
                        min_mad = mad

                if tmp_move_x != 0 or tmp_move_y != 0:
                    move_x += tmp_move_x
                    move_y += tmp_move_y
                else:
                    break

            tmp_move_x = 0
            tmp_move_y = 0
            for p in range(-1, 2):
                for q in range(-1, 2):
                    if (x + search_range + move_x + p < x + 2 * search_range and x + search_range + move_x + p >= x and
                            y + search_range + move_y + q < y + 2 * search_range and y + search_range + move_y + q >= y):

                        current_block = current_frame[x:x + block_size, y:y + block_size]
                        ref_block = padding_reference_frame[
                                    x + search_range + move_x + p: x + search_range + move_x + p + block_size,
                                    y + search_range + move_y + q: y + search_range + move_y + q + block_size]

                        mad = np.sum(np.abs(current_block - ref_block))

                        if min_mad > mad:
                            tmp_move_x = p
                            tmp_move_y = q
                            min_mad = mad

            vector[i, j, 0] = move_x + tmp_move_x
            vector[i, j, 1] = move_y + tmp_move_y

    motion = vector
    return motion

frame_number = [91]
seq1 = []

for i in range(frame_number[0]):
    filename = f"./s1/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq1.append(image)

seq1 = np.stack(seq1, axis=-1)

frame_number2 = [30]
seq2 = []

for i in range(1, frame_number2[0]+1):
    if i < 10:
        filename = f"./s2/0{i}.bmp"
    else:
        filename = f"./s2/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq2.append(image)

seq2 = np.stack(seq2, axis=-1)
start = time.time()

# Full search
print("Do 2D logarithm search algorithm")

# seq1 search range1
search_range = [8]  # Change this value according to your search_range
block_size = 8

search_range1_seq1_reconstruct_full_search = np.copy(seq1)
search_range1_seq1_psnr_full_search = np.zeros(frame_number[0])

for f in range(frame_number[0] - 1):
    motion = two_d_logarithm_search(search_range1_seq1_reconstruct_full_search[..., f], seq1[..., f + 1], search_range[0])
    block_x_num, block_y_num, x_y = motion.shape
    padding_reference_frame = cv2.copyMakeBorder(search_range1_seq1_reconstruct_full_search[..., f], search_range[0], search_range[0], search_range[0], search_range[0], cv2.BORDER_REPLICATE)
    padding_reference_frame = padding_reference_frame.astype(np.float64)

    for i in range(block_x_num):
        for j in range(block_y_num):
            search_range1_seq1_reconstruct_full_search[(i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size), f + 1] = padding_reference_frame[search_range[0] + (i * block_size) + int(motion[i, j, 0]):search_range[0] + ((i + 1) * block_size) + int(motion[i, j, 0]), search_range[0] + (j * block_size) + int(motion[i, j, 1]):search_range[0] + ((j + 1) * block_size) + int(motion[i, j, 1])]

for i in range(frame_number[0] - 1):
    search_range1_seq1_psnr_full_search[i] = cv2.PSNR(search_range1_seq1_reconstruct_full_search[..., i+1], seq1[..., i+1])

output_dir = 's1_2d_logarithm_search_range_8_size_8_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, frame_number[0]):
    output_filename = os.path.join(output_dir, f"reconstructed_frame_{i}.bmp")
    cv2.imwrite(output_filename, search_range1_seq1_reconstruct_full_search[..., i])

with open("s1_2d_logarithm_search_range_8_size_8_output/psnr_list.txt", 'w') as text_file:
    for i in range(0, frame_number[0] - 1):
        text_file.write(f"Reconstructed frame {i+1} psnr: {search_range1_seq1_psnr_full_search[i]}\n")

def three_step_search(reference_frame, current_frame, search_range):
    m, n = reference_frame.shape
    block_size = 16
    padding_reference_frame = cv2.copyMakeBorder(reference_frame, search_range, search_range, search_range, search_range, cv2.BORDER_REPLICATE)
    vector = np.zeros((m // block_size, n // block_size, 2))
    padding_reference_frame = padding_reference_frame.astype(np.float64)
    current_frame = current_frame.astype(np.float64)
    for i in range(m // block_size):
        for j in range(n // block_size):
            x = i * block_size
            y = j * block_size
            move_x = 0
            move_y = 0
            min_mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] -
                                     padding_reference_frame[x + search_range:x + block_size + search_range,
                                                              y + search_range:y + block_size + search_range]))
            step_range = 4
            while step_range >= 1:
                tmp_move_x = 0
                tmp_move_y = 0
                for p in range(-step_range, step_range + 1, step_range):
                    for q in range(-step_range, step_range + 1, step_range):
                        if x + search_range + move_x + p < x + 2 * search_range and x + search_range + move_x + p >= x and \
                                y + search_range + move_y + q < y + 2 * search_range and y + search_range + move_y + q >= y:
                            mad = np.sum(np.abs(current_frame[x:x + block_size, y:y + block_size] -
                                                padding_reference_frame[
                                                x + search_range + move_x + p:x + block_size + search_range + move_x + p,
                                                y + search_range + move_y + q:y + block_size + search_range + move_y + q]))
                            if min_mad > mad:
                                tmp_move_x = p
                                tmp_move_y = q
                                min_mad = mad
                move_x += tmp_move_x
                move_y += tmp_move_y
                step_range //= 2
            vector[i, j, 0] = move_x
            vector[i, j, 1] = move_y
    motion = vector
    return motion


frame_number = [91]
seq1 = []

for i in range(frame_number[0]):
    filename = f"./s1/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq1.append(image)

seq1 = np.stack(seq1, axis=-1)

frame_number2 = [30]
seq2 = []

for i in range(1, frame_number2[0]+1):
    if i < 10:
        filename = f"./s2/0{i}.bmp"
    else:
        filename = f"./s2/{i}.bmp"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    seq2.append(image)

seq2 = np.stack(seq2, axis=-1)
start = time.process_time()

# Full search
print("Do 3-step search algorithm")

# seq1 search range1
search_range = [8]  # Change this value according to your search_range
block_size = 8

search_range1_seq1_reconstruct_full_search = np.copy(seq1)
search_range1_seq1_psnr_full_search = np.zeros(frame_number[0])

for f in range(frame_number[0] - 1):
    motion = three_step_search(search_range1_seq1_reconstruct_full_search[..., f], seq1[..., f + 1], search_range[0])
    block_x_num, block_y_num, x_y = motion.shape
    padding_reference_frame = cv2.copyMakeBorder(search_range1_seq1_reconstruct_full_search[..., f], search_range[0], search_range[0], search_range[0], search_range[0], cv2.BORDER_REPLICATE)
    padding_reference_frame = padding_reference_frame.astype(np.float64)

    for i in range(block_x_num):
        for j in range(block_y_num):
            search_range1_seq1_reconstruct_full_search[(i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size), f + 1] = padding_reference_frame[search_range[0] + (i * block_size) + int(motion[i, j, 0]):search_range[0] + ((i + 1) * block_size) + int(motion[i, j, 0]), search_range[0] + (j * block_size) + int(motion[i, j, 1]):search_range[0] + ((j + 1) * block_size) + int(motion[i, j, 1])]

for i in range(frame_number[0] - 1):
    search_range1_seq1_psnr_full_search[i] = cv2.PSNR(search_range1_seq1_reconstruct_full_search[..., i+1], seq1[..., i+1])

output_dir = 's1_3-step_search_range_8_size_8_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, frame_number[0]):
    output_filename = os.path.join(output_dir, f"reconstructed_frame_{i}.bmp")
    cv2.imwrite(output_filename, search_range1_seq1_reconstruct_full_search[..., i])

with open("s1_3-step_search_range_8_size_8_output/psnr_list.txt", 'w') as text_file:
    for i in range(0, frame_number[0] - 1):
        text_file.write(f"Reconstructed frame {i+1} psnr: {search_range1_seq1_psnr_full_search[i]}\n")