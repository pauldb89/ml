import itertools
import os
from argparse import ArgumentParser

import torch
from torchvision import transforms

from neural_style_transfer.consts import COCO_EVAL
from neural_style_transfer.consts import WIKIART_EVAL
from neural_style_transfer.data import create_eval_data_loader
from neural_style_transfer.model import StyleTransfer


def main():
    torch.distributed.init_process_group("nccl")
    parser = ArgumentParser()
    parser.add_argument("--content_dir", type=str, default=COCO_EVAL, help="Content images directory")
    parser.add_argument("--style_dir", type=str, default=WIKIART_EVAL, help="Style images directory")
    parser.add_argument("--model_snapshot", type=str, required=True, help="Model snapshot")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_loader = create_eval_data_loader(
        content_root_dir=args.content_dir,
        style_root_dir=args.style_dir,
        batch_size=1,
        num_workers=1,
    )

    model = StyleTransfer(
        content_feature_name="layer4",
        style_feature_names=["layer1", "layer2", "layer3", "layer4"],
    )
    model.load_state_dict(torch.load(args.model_snapshot, map_location="cpu"))
    model.cuda()
    model.eval()

    image_transform = transforms.ToPILImage()
    for idx, (content_image, style_image) in enumerate(itertools.islice(data_loader, args.num_examples)):
        tensor = model.eval_forward(content_image, style_image)
        image = image_transform(torch.squeeze(tensor, dim=0))
        image.save(os.path.join(args.output_dir, f"{idx}.png"))


if __name__ == "__main__":
    main()
