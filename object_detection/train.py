from torchvision.datasets import CocoDetection

from object_classification.distributed_utils import print_once


def main():
    train_dataset = CocoDetection(
        root="/datasets/coco/train2017",
        annFile="/datasets/coco/annotations/instances_train2017.json",
    )
    print_once(f"Training dataset has {len(train_dataset)} examples")

    eval_dataset = CocoDetection(
        root="/datasets/coco/val2017",
        annFile="/datasets/coco/annotations/instances_val2017.json",
    )
    print_once(f"Eval dataset has {len(eval_dataset)} examples")


if __name__ == "__main__":
    main()
