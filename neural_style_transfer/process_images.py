import glob
import hashlib
import os
import shutil

import tqdm

SOURCE_DIR = "/datasets/wikiart/raw"
TRAIN_DIR = "/local_datasets/wikiart/train"
TEST_DIR = "/local_datasets/wikiart/test"


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    for idx, filename in tqdm.tqdm(list(enumerate(glob.glob(f"{SOURCE_DIR}/*/*")))):
        prefix = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:10]
        new_filename = f"{prefix}_{os.path.basename(filename)}"
        if idx % 10 == 0:
            shutil.copyfile(filename, os.path.join(TEST_DIR, new_filename))
        else:
            shutil.copyfile(filename, os.path.join(TRAIN_DIR, new_filename))


if __name__ == "__main__":
    main()
