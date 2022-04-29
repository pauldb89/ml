import os
from argparse import ArgumentParser

import torch.distributed
import torchvision
from torch.nn.parallel import DistributedDataParallel

from object_classification.data import get_eval_data_loader
from object_classification.distributed_utils import print_once
from object_classification.distributed_utils import world_size
from object_classification.evaluate import evaluate
from object_classification.models.model_wrapper import PretrainedWrapper


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Which dataset to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    args = parser.parse_args()

    data_loader = get_eval_data_loader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print_once(f"Eval dataset has {len(data_loader.dataset) * world_size()} examples")

    torch.hub.set_dir(f"/root/.cache/torch/{local_rank}")
    model = torchvision.models.vgg16_bn(pretrained=True)
    model = PretrainedWrapper(model)
    model.cuda()

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    evaluate(step=0, model=model, data_loader=data_loader)


if __name__ == "__main__":
    main()
