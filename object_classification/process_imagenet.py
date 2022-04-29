import os
from argparse import ArgumentParser
import xml.etree.ElementTree as ET

from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--image_dir", required=str, help="Image directory")
    parser.add_argument("--label_dir", required=str, help="Label directory")
    parser.add_argument("--output_dir", required=str, help="Output directory")
    args = parser.parse_args()

    seen_labels = set()

    image_filenames = sorted(os.listdir(args.image_dir))
    label_filenames = sorted(os.listdir(args.label_dir))
    assert len(image_filenames) == len(label_filenames), f"{len(image_filenames)} {len(label_filenames)}"

    for image_filename, label_filename in tqdm(list(zip(image_filenames, label_filenames))):
        assert image_filename.split(".")[0] == label_filename.split(".")[0], f"{image_filename} {label_filename}"

        tree = ET.parse(os.path.join(args.label_dir, label_filename))
        label = tree.find(".//name").text

        target_dir = os.path.join(args.output_dir, label)
        if label not in seen_labels:
            os.makedirs(target_dir, exist_ok=True)
            seen_labels.add(label)

        os.rename(os.path.join(args.image_dir, image_filename), os.path.join(target_dir, image_filename))


if __name__ == "__main__":
    main()
