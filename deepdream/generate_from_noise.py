import functools
from argparse import ArgumentParser
from typing import Callable
from typing import Dict

import PIL
from lucent.modelzoo import inceptionv1
import torch
import torchvision.models
from PIL import Image
from lucent.optvis import transform
from torch import nn
from torchvision.transforms import transforms

from common.consts import IMAGENET_MEAN
from common.consts import IMAGENET_STD

MODEL_LOADERS = {
    "inception_v3": lambda: torchvision.models.inception_v3(pretrained=True, aux_logits=False),
    "inception_v1": lambda: inceptionv1(pretrained=True),
}


def load_model(model_name: str) -> nn.Module:
    """
    Load a pretrained torchvision model.
    """
    assert model_name in MODEL_LOADERS, f"Model {model_name} cannot be loaded"

    model = MODEL_LOADERS[model_name]()
    model.eval()
    return model


def track_activations(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Add hooks to track activations on each of the model's modules.
    """
    activations = {}
    for name, module in model.named_modules():
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor, module_name: str) -> None:
            activations[module_name] = output

        module.register_forward_hook(functools.partial(hook, module_name=name))

    return activations


def compute_loss(objectives: str, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for objective in objectives.split(","):
        print(objective)
        layer_name, sign, channel, neuron_id = objective.split(":")
        activation = torch.squeeze(activations[layer_name], dim=0)
        if channel:
            channel = int(channel)
            assert 0 <= channel < activation.size()[0]
            activation = activation[channel]
        if neuron_id:
            neuron_id = int(neuron_id)
            assert 0 <= neuron_id < torch.numel(activation)
            activation = activation.view(-1)[int(neuron_id)]
        assert sign in {"0", "1"}
        losses.append(-torch.mean(activation) if sign == "1" else torch.mean(activation))
    return torch.sum(torch.stack(losses))


def create_loss_fn(
    objectives: str,
    activations: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def loss_fn(image_tensor: torch.Tensor) -> torch.Tensor:
        model(image_tensor)
        return compute_loss(objectives=objectives, activations=activations)

    return loss_fn


def to_image(image_tensor: torch.Tensor) -> PIL.Image.Image:
    image_tensor = torch.squeeze(image_tensor, dim=0)
    image_tensor = torch.permute(image_tensor, dims=[1, 2, 0])
    image_tensor = (image_tensor * 255).to(torch.int8)
    return PIL.Image.fromarray(image_tensor.cpu().numpy(), mode='RGB')


class InceptionV1Transform(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 255 - 127


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=512, help="Number of iterations")
    parser.add_argument("--display_every_n_steps", type=int, default=128, help="Display image every n steps")
    parser.add_argument("--output_filename", type=str, default=None, help="Output file")
    parser.add_argument(
        "--objectives",
        type=str,
        default="mixed4a:1:476:",
        help="Comma separated list of layer:sign:filter:neuron to excite",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="inception_v1",
        choices=MODEL_LOADERS.keys(),
        help="What model to use",
    )
    args = parser.parse_args()

    model = load_model(args.model_name)
    model.to(device)

    activations = track_activations(model)
    loss_fn = create_loss_fn(args.objectives, activations, model)

    image_tensor = torch.randn(1, 3, 244, 244, device=device) * 0.01
    image_tensor.requires_grad = True

    # Apply sigmoid as a transform on the image to produce valid image values scaled between [0, 1].
    image_transform_fn = torch.sigmoid

    # Apply normalization, etc. to make the input image compatible with the space on which the model was trained.
    model_transform_fns = transform.standard_transforms.copy()
    # Note(pauldb): Using the transforms from the lucid codebase is critical for obtaining robust, good looking images.
    if args.model_name == "inception_v1":
        model_transform_fns.append(InceptionV1Transform())
    else:
        model_transform_fns.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    model_transform_fn = transforms.Compose(model_transform_fns)

    optimizer = torch.optim.Adam([image_tensor], lr=5e-2)
    for step in range(args.steps):
        optimizer.zero_grad()
        loss = loss_fn(model_transform_fn(image_transform_fn(image_tensor)))
        print(f"Step {step} Loss {loss.item()}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if 0 < step < args.steps and step % args.display_every_n_steps == 0:
                to_image(image_transform_fn(image_tensor)).show()

    image = to_image(image_transform_fn(image_tensor))
    if args.output_filename:
        image.save(args.output_filename)
    else:
        image.show()


if __name__ == "__main__":
    main()
