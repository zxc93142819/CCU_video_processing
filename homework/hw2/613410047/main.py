import cv2
import numpy as np
import os
import time

def load_sequence(folder_path: str):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    return [cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE) for f in image_files]

def full_search(ref, target, block_size, search_range):
    h, w = ref.shape
    padded_ref = cv2.copyMakeBorder(ref, search_range, search_range, search_range, search_range, cv2.BORDER_REPLICATE)

    padded_ref = padded_ref.astype(np.float64)
    target = target.astype(np.float64)

    mv = np.zeros((h // block_size, w // block_size, 2))
    start = time.time()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            min_error = float('inf')
            best_mv = (0, 0)
            block = target[i:i+block_size, j:j+block_size]
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    y_pos = search_range + i + dy
                    x_pos = search_range + j + dx
                    # if (0 <= y_pos) and (0 <= x_pos) and (y_pos + block_size <= padded_ref.shape[0]) and (x_pos + block_size <= padded_ref.shape[1]):
                    ref_block = padded_ref[y_pos:y_pos+block_size, x_pos:x_pos+block_size]
                    error = np.sum(np.abs(block - ref_block))
                    if error < min_error:
                        min_error = error
                        best_mv = (dy, dx)
            mv[i//block_size, j//block_size] = best_mv

    return mv, time.time() - start

def log_search(ref, target, block_size, search_range):
    h, w = ref.shape
    search_range = search_range
    padded_ref = cv2.copyMakeBorder(ref, search_range, search_range, search_range, search_range, cv2.BORDER_REPLICATE)

    padded_ref = padded_ref.astype(np.float64)
    target = target.astype(np.float64)

    mv = np.zeros((h // block_size, w // block_size, 2))
    start = time.time()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = target[i:i+block_size, j:j+block_size]
            step = 2 ** int(np.floor(np.log2(search_range)))
            original_center = [i + search_range , j + search_range]
            center = [i + search_range , j + search_range]

            while step >= 1:
                min_error = float('inf')
                best_mv = [0, 0]
                dy = [0 , -step , 0 , 0 , step]
                dx = [0 , 0 , -step , step , 0]
                for kk in range(len(dy)) :
                    y_pos = center[0] + dy[kk]
                    x_pos = center[1] + dx[kk]
                    # print(center[0] , center[1] , y_pos , x_pos , search_range , step)
                    if (0 <= y_pos) and (0 <= x_pos) and (y_pos + block_size <= padded_ref.shape[0]) and (x_pos + block_size <= padded_ref.shape[1]):
                        ref_block = padded_ref[y_pos:y_pos+block_size, x_pos:x_pos+block_size]
                        error = np.sum(np.abs(block - ref_block))
                        if error < min_error:
                            min_error = error
                            best_mv = [dy[kk], dx[kk]]
                center[0] += best_mv[0]
                center[1] += best_mv[1]
                step //= 2

            mv[i//block_size, j//block_size] = [center[0] - original_center[0] , center[1] - original_center[1]]

    return mv, time.time() - start

def three_step_search(ref, target, block_size, search_range):
    h, w = ref.shape
    search_range = search_range
    padded_ref = cv2.copyMakeBorder(ref, search_range, search_range, search_range, search_range, cv2.BORDER_REPLICATE)

    padded_ref = padded_ref.astype(np.float64)
    target = target.astype(np.float64)

    mv = np.zeros((h // block_size, w // block_size, 2))
    start = time.time()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = target[i:i+block_size, j:j+block_size]
            step = search_range // 2
            original_center = [i + search_range , j + search_range]
            center = [i + search_range , j + search_range]
            min_error = float('inf')

            while step >= 1:
                best_mv = [0, 0]
                dy = [0 , -step , -step , -step , 0 , 0 , 0 , step , step , step]
                dx = [0 , -step , 0 , step , -step , 0 , step , -step , 0 , step]
                for kk in range(len(dy)) :
                    y_pos = center[0] + dy[kk]
                    x_pos = center[1] + dx[kk]
                    # print(center[0] , center[1] , y_pos , x_pos , search_range , step)
                    if (0 <= y_pos) and (0 <= x_pos) and (y_pos + block_size <= padded_ref.shape[0]) and (x_pos + block_size <= padded_ref.shape[1]):
                        ref_block = padded_ref[y_pos:y_pos+block_size, x_pos:x_pos+block_size]
                        error = np.sum(np.abs(block - ref_block))
                        if error < min_error:
                            min_error = error
                            best_mv = [dy[kk], dx[kk]]
                center[0] += best_mv[0]
                center[1] += best_mv[1]
                step //= 2

            mv[i//block_size, j//block_size] = [center[0] - original_center[0] , center[1] - original_center[1]]

    return mv, time.time() - start

def reconstruct_frame(ref_frame, motion_vectors, block_size , search_range):
    recon = np.copy(ref_frame)
    padded_ref = cv2.copyMakeBorder(ref_frame, search_range, search_range, search_range, search_range, cv2.BORDER_REPLICATE)
    padded_ref = padded_ref.astype(np.float64)
    h , w = ref_frame.shape

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            mv = motion_vectors[i // block_size, j // block_size]
            ref_y = search_range + i + int(mv[0])
            ref_x = search_range + j + int(mv[1])

            recon[i:i + block_size , j:j + block_size] = padded_ref[ref_y:ref_y + block_size , ref_x:ref_x + block_size]

    return recon

def calculate_psnr(original, reconstructed):
    return cv2.PSNR(original , reconstructed)

def all_algorithms(folder_path, block_sizes=[8, 16], search_ranges=[8, 16]):
    seq = ["s1" , "s2"]
    seq_dir_path = ["s1" , "s2"]
    method_dir_path = ["full_search" , "2d_log_search" , "three_step_search"]
    for seq_index in range(len(seq)) :
        seqq = seq[seq_index]
        seq_path = folder_path + seqq
        frames = load_sequence(seq_path)

        for block_size in block_sizes:
            for search_range in search_ranges:
                block_search_dir_path = f"search_range_{search_range}_block_size_{block_size}"
                print(f"\n==== Sequence: {seqq} | Block size: {block_size} | Search range: {search_range} ====")
                
                total_time = [0 , 0 , 0]
                psnr_info = ["" , "" , ""]
                output_dir = ["" , "" , ""]
                recon = frames[0]
                for i in range(len(frames) - 1) :
                    ref_frame = recon
                    target_frame = frames[i + 1]

                    # ---------------------------------------------------------------------------------------------------------------
                    # full search
                    mv, t = full_search(ref_frame, target_frame, block_size, search_range)
                    recon = reconstruct_frame(ref_frame, mv, block_size , search_range)
                    psnr = calculate_psnr(target_frame, recon)
                    total_time[0] += t

                    # output
                    temp = os.path.join(seq_dir_path[seq_index] , method_dir_path[0])
                    output_dir[0] = os.path.join(temp , block_search_dir_path)
                    if not os.path.exists(output_dir[0]):
                        os.makedirs(output_dir[0])

                    output_filename = os.path.join(output_dir[0], f"reconstructed_frame_{i + 1}.bmp")
                    cv2.imwrite(output_filename, recon)

                    psnr_info[0] += f"Reconstructed frame {i + 1} psnr: {psnr}\n"

                recon = frames[0]
                # ============================================================================================
                for i in range(len(frames) - 1) :
                    ref_frame = recon
                    target_frame = frames[i + 1]

                    # ---------------------------------------------------------------------------------------------------------------
                    # 2-D log
                    mv, t = log_search(ref_frame, target_frame, block_size, search_range)
                    recon = reconstruct_frame(ref_frame, mv, block_size , search_range)
                    psnr = calculate_psnr(target_frame, recon)
                    total_time[1] += t

                    # output
                    temp = os.path.join(seq_dir_path[seq_index] , method_dir_path[1])
                    output_dir[1] = os.path.join(temp , block_search_dir_path)
                    if not os.path.exists(output_dir[1]):
                        os.makedirs(output_dir[1])

                    output_filename = os.path.join(output_dir[1], f"reconstructed_frame_{i + 1}.bmp")
                    cv2.imwrite(output_filename, recon)
                    
                    psnr_info[1] += f"Reconstructed frame {i + 1} psnr: {psnr}\n"

                recon = frames[0]
                # ===========================================================================================================
                for i in range(len(frames) - 1) :
                    ref_frame = recon
                    target_frame = frames[i + 1]

                    # ---------------------------------------------------------------------------------------------------------------
                    # three-step search
                    mv, t = three_step_search(ref_frame, target_frame, block_size, search_range)
                    recon = reconstruct_frame(ref_frame , mv, block_size , search_range)
                    psnr = calculate_psnr(target_frame, recon)
                    total_time[2] += t

                    # output
                    temp = os.path.join(seq_dir_path[seq_index] , method_dir_path[2])
                    output_dir[2] = os.path.join(temp , block_search_dir_path)
                    if not os.path.exists(output_dir[2]):
                        os.makedirs(output_dir[2])

                    output_filename = os.path.join(output_dir[2], f"reconstructed_frame_{i + 1}.bmp")
                    cv2.imwrite(output_filename, recon)

                    psnr_info[2] += f"Reconstructed frame {i + 1} psnr: {psnr}\n"
                
                    # ---------------------------------------------------------------------------------------------------------------

                with open(output_dir[0] + "psnr_list.txt", 'w') as text_file:
                    text_file.write(psnr_info[0])
                with open(output_dir[1] + "psnr_list.txt", 'w') as text_file:
                    text_file.write(psnr_info[1])
                with open(output_dir[2] + "psnr_list.txt", 'w') as text_file:
                    text_file.write(psnr_info[2])
                print(f"full Search       Time: {total_time[0]:.4f} seconds")
                print(f"2D Log Search     Time: {total_time[1]:.4f} seconds")
                print(f"3-Step Search     Time: {total_time[2]:.4f} seconds")

all_algorithms("./HW2_test_sequence/")