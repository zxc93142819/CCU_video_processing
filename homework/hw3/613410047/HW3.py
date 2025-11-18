import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def load_frames(folder):
    frames = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            frames.append(img)
    return frames

def h263_deblocking_filter(frame, block_size=8, QP=40):
    filtered = frame.copy().astype(np.float64)
    h, w = frame.shape

    for i in range(block_size, h - block_size + 1, block_size):
        for j in range(block_size, w - block_size + 1, block_size):
            for k in range(block_size):
                # Vertical deblocking
                A = filtered[i + k, j - 2]
                B = filtered[i + k, j - 1]
                C = filtered[i + k, j]
                D = filtered[i + k, j + 1]
                d = (3 * A - 8 * B + 8 * C - 3 * D) / 16
                d1 = np.sign(d) * (np.maximum(0, np.abs(d) - np.maximum(0, 2 * np.abs(d) - QP)))
                filtered[i + k, j - 1] = np.uint8(np.clip(B + d1, 0, 255))
                filtered[i + k, j] = np.uint8(np.clip(C - d1, 0, 255))

                # Horizontal deblocking
                A = int(filtered[i - 2, j + k])
                B = int(filtered[i - 1, j + k])
                C = int(filtered[i, j + k])
                D = int(filtered[i + 1, j + k])
                d = (3 * A - 8 * B + 8 * C - 3 * D) / 16
                d1 = np.sign(d) * (np.maximum(0, np.abs(d) - np.maximum(0, 2 * np.abs(d) - QP)))
                filtered[i - 1, j + k] = np.uint8(np.clip(B + d1, 0, 255))
                filtered[i, j + k] = np.uint8(np.clip(C - d1, 0, 255))

    return filtered

def h264_deblocking_filter(frame, block_size=4, alpha=60, beta=30):
    # Simple H.264 deblocking: more sophisticated, uses thresholds alpha/beta
    filtered = frame.copy().astype(np.float64)
    h, w = frame.shape
    block_size = 4

    for i in range(block_size, h - block_size + 1, block_size):
        for j in range(block_size, w - block_size + 1, block_size):
            for k in range(block_size):
                p3 = filtered[i + k, j - 4]
                p2 = filtered[i + k, j - 3]
                p1 = filtered[i + k, j - 2]
                p0 = filtered[i + k, j - 1]
                q0 = filtered[i + k, j]
                q1 = filtered[i + k, j + 1]
                q2 = filtered[i + k, j + 2]
                q3 = filtered[i + k, j + 3]
                ap = np.abs(p2 - p0)
                aq = np.abs(q2 - q0)

                # Left/upper side
                if ap < beta and np.abs(p0 - q0) < ((alpha // 2) + 2):
                    filtered[i + k, j - 1] = np.uint8((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) // 8)
                    filtered[i + k, j - 2] = np.uint8((p2 + p1 + p0 + q0 + 2) // 4)
                    filtered[i + k, j - 3] = np.uint8((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) // 8)
                else:
                    filtered[i + k, j - 1] = np.uint8((2 * p1 + p0 + q1 + 2) // 4)
                
                # Right/lower side
                if aq < beta and np.abs(p0 - q0) < ((alpha // 2) + 2):
                    filtered[i + k, j] = np.uint8((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) // 8)
                    filtered[i + k, j + 1] = np.uint8((p0 + q0 + q1 + q2 + 2) // 4)
                    filtered[i + k, j + 2] = np.uint8((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) // 8)
                else:
                    filtered[i + k, j] = np.uint8((2 * q1 + q0 + p1 + 2) // 4)

                # Vertical deblocking
                p3 = filtered[i - 4, j + k]
                p2 = filtered[i - 3, j + k]
                p1 = filtered[i - 2, j + k]
                p0 = filtered[i - 1, j + k]
                q0 = filtered[i, j + k]
                q1 = filtered[i + 1, j + k]
                q2 = filtered[i + 2, j + k]
                q3 = filtered[i + 3, j + k]
                ap = np.abs(p2 - p0)
                aq = np.abs(q2 - q0)

                # Left/upper side
                if ap < beta and np.abs(p0 - q0) < ((alpha // 2) + 2):
                    filtered[i - 1, j + k] = np.uint8((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) // 8)
                    filtered[i - 2, j + k] = np.uint8((p2 + p1 + p0 + q0 + 2) // 4)
                    filtered[i - 3, j + k] = np.uint8((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) // 8)
                else:
                    filtered[i - 1, j + k] = np.uint8((2 * p1 + p0 + q1 + 2) // 4)

                # Right/lower side
                if aq < beta and np.abs(p0 - q0) < ((alpha // 2) + 2):
                    filtered[i , j + k] = np.uint8((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) // 8)
                    filtered[i + 1, j + k] = np.uint8((p0 + q0 + q1 + q2 + 2) // 4)
                    filtered[i + 2, j + k] = np.uint8((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) // 8)
                else:
                    filtered[i , j + k] = np.uint8((2 * q1 + q0 + p1 + 2) // 4)

    return filtered

def main():
    frames = load_frames('Hw3_test sequences/Holmes/Original')
    decompressed = load_frames('Hw3_test sequences/Holmes/Decompressed')

    psnr_h263_4 = []
    psnr_h263_8 = []
    psnr_h264 = []
    psnr_decompressed = []

    for idx, frame in enumerate(frames):
        # Add artificial blocking for demonstration (skip if not needed)
        # blocked = frame

        # H.263
        # block_size = 8
        filtered_h263_8 = h263_deblocking_filter(decompressed[idx] , block_size = 8)
        # block_size = 4
        filtered_h263_4 = h263_deblocking_filter(decompressed[idx] , block_size = 4)
        psnr263_4 = compare_psnr(frame, filtered_h263_4, data_range=255)
        psnr263_8 = compare_psnr(frame, filtered_h263_8, data_range=255)
        psnr_h263_4.append(psnr263_4)
        psnr_h263_8.append(psnr263_8)

        # output
        output_dir = os.path.join("output_H263" , "block_size=4")  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, f"deblocking_frame_{idx + 1}.bmp")
        cv2.imwrite(output_filename, filtered_h263_4)

        output_dir = os.path.join("output_H263" , "block_size=8")  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, f"deblocking_frame_{idx + 1}.bmp")
        cv2.imwrite(output_filename, filtered_h263_8)

        # H.264
        filtered_h264 = h264_deblocking_filter(decompressed[idx])
        psnr264 = compare_psnr(frame, filtered_h264, data_range=255)
        psnr_h264.append(psnr264)

        output_dir = "output_H264"  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, f"deblocking_frame_{idx + 1}.bmp")
        cv2.imwrite(output_filename, filtered_h264)

        # decompressed
        psnr_de = compare_psnr(frame, decompressed[idx], data_range=255)
        psnr_decompressed.append(psnr_de)

        print(f"Frame {idx + 1}: H.263 PSNR (block size 4)={psnr263_4:.2f}, H.263 PSNR (block size 8)={psnr263_8:.2f}, H.264 PSNR={psnr264:.2f}, Decompressed PSNR={psnr_de:.2f}")

    print(f"Average H.263 PSNR (block size 4): {np.mean(psnr_h263_4):.2f}")
    print(f"Average H.263 PSNR (block size 8): {np.mean(psnr_h263_8):.2f}")
    print(f"Average H.264 PSNR: {np.mean(psnr_h264):.2f}")
    print(f"Average Decompressed PSNR: {np.mean(psnr_decompressed):.2f}")

if __name__ == "__main__":
    main()