import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

def get_roi_by_mask(image, mask):
    # mask should be binary, extract non-zero region
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def process_and_compare(base_dir, ct_modality, test_method):
    label_dir = f'{base_dir}/LABEL/test/tumor'
    test_dir = f'{base_dir}/{ct_modality}/test/tumor'
    method_dir = f'{base_dir}/{ct_modality}/{test_method}/tumor'

    psnr_list = []
    ssim_list = []

    for filename in os.listdir(label_dir):
        base_name = '_'.join(filename.split('_')[:-1])
        label_path = os.path.join(label_dir, filename)
        mask = load_image(label_path)
        if mask is None:
            continue

        test_img_path = os.path.join(test_dir, base_name + '.png')
        method_img_path = os.path.join(method_dir, base_name + '.png')
        
        test_img = load_image(test_img_path)
        method_img = load_image(method_img_path)

        if test_img is None or method_img is None:
            print(f"Missing file for base {base_name}, skipping.")
            continue

        # only compare within mask region
        test_roi, _ = get_roi_by_mask(test_img, mask)
        method_roi, _ = get_roi_by_mask(method_img, mask)

        p = psnr(test_roi, method_roi)
        s = ssim(test_roi, method_roi)

        psnr_list.append(p)
        ssim_list.append(s)
        print(f"{filename} - PSNR: {p:.2f}, SSIM: {s:.4f}")

    if psnr_list and ssim_list:
        print(f"Average PSNR: {np.mean(psnr_list):.2f}, Average SSIM: {np.mean(ssim_list):.4f}")
    else:
        print("No valid samples found.")

def main():
    parser = argparse.ArgumentParser(description="Compute average PSNR/SSIM for dataset.")
    parser.add_argument('--base_dir', type=str, required=True, help='Path to base_dir')
    parser.add_argument('--ct_modality', type=str, required=True, help='CT modality (CT or CECT)')
    parser.add_argument('--test_method', type=str, required=True, help='Test method folder name')
    args = parser.parse_args()
    process_and_compare(args.base_dir, args.ct_modality, args.test_method)

if __name__ == '__main__':
    main()
