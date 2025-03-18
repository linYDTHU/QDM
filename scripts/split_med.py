"""
Split images into train/test sets
"""

import os
import argparse
import shutil
from sklearn.model_selection import train_test_split

def main():
    # set up argument parser
    parser = argparse.ArgumentParser(description='Split images into train/test sets')
    parser.add_argument('--source', required=True, nargs='+',
                        help='Source directories containing image folders')
    parser.add_argument('--target', required=True,
                        help='Target directory to save train/test sets')
    args = parser.parse_args()

    # create target directories
    os.makedirs(os.path.join(args.target, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.target, 'test'), exist_ok=True)

    # collect image files
    image_files = []
    for source_dir in args.source:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    subdir_name = os.path.basename(root)
                    new_filename = f"{subdir_name}_{file}"
                    src_path = os.path.join(root, file)
                    image_files.append((src_path, new_filename))

    # split image files into train/test sets
    train_files, test_files = train_test_split(
        image_files, test_size=0.05, random_state=42
    )

    # copy files to target directories
    def copy_files(file_list, target_subdir):
        for src_path, new_filename in file_list:
            dst_path = os.path.join(args.target, target_subdir, new_filename)
            shutil.copy2(src_path, dst_path)

    # execute copying
    copy_files(train_files, 'train')
    copy_files(test_files, 'test')

    print(f"\nFinished processing {len(image_files)} images")
    print(f"Source directories: {args.source}")
    print(f"Target directory: {args.target}")
    print(f"Trainset: {len(train_files)} images")
    print(f"Testset: {len(test_files)} images")

if __name__ == "__main__":
    main()