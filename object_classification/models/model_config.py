from dataclasses import dataclass


@dataclass
class ModelConfig:
    train_resize_dim: int
    eval_resize_dim: int
    eval_crop_dim: int
    use_batch_norm: bool


MODEL_CONFIGS = {
    "vgg16": ModelConfig(train_resize_dim=224, eval_resize_dim=256, eval_crop_dim=224, use_batch_norm=False),
    "vgg16bn": ModelConfig(train_resize_dim=224, eval_resize_dim=256, eval_crop_dim=224, use_batch_norm=True),
    "inception_v3": ModelConfig(train_resize_dim=299, eval_resize_dim=299, eval_crop_dim=299, use_batch_norm=True),
    "resnet50": ModelConfig(train_resize_dim=224, eval_resize_dim=256, eval_crop_dim=224, use_batch_norm=True),
}
