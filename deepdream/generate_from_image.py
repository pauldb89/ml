from argparse import ArgumentParser
from typing import Callable
from typing import Dict
import torch
import torchvision.models
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
import tensorflow as tf

DEFAULT_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
DEFAULT_MODULES = "Mixed_6a,Mixed_6c"


def get_image(url: str, max_dim: int = 500) -> Image.Image:
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = Image.open(image_path)
    img.thumbnail((max_dim, max_dim))
    return img


def compute_loss(
    image: torch.Tensor,
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    class_id: int,
) -> torch.Tensor:
    predictions = model(image)
    if class_id != -1:
        return torch.sum(predictions[:, class_id])

    losses = []
    for activation in activations.values():
        losses.append(torch.mean(activation))

    return -torch.sum(torch.stack(losses))


def main():
    parser = ArgumentParser()
    parser.add_argument("--image_url", type=str, default=DEFAULT_URL, help="Image URL to process")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--module_names", type=str, default=DEFAULT_MODULES, help="List of modules to excite")
    parser.add_argument("--class_id", type=int, default=-1, help="Class outputs to excite")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--display_every_n_steps", type=int, default=200, help="Show updated image every N steps")
    parser.add_argument("--resolution_scale", type=float, default=1.3, help="Resolution (aka octave) scale")
    parser.add_argument("--min_resolution", type=int, default=0, help="Min resolution exponent")
    parser.add_argument("--max_resolution", type=int, default=1, help="Max resolution exponent")
    args = parser.parse_args()

    image = get_image(args.image_url)

    to_tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )
    to_image_transform = transforms.Compose([
        transforms.Normalize(
            mean=-1,
            std=2,
        ),
        transforms.ToPILImage(),
    ])

    model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    model.eval()
    module_names = args.module_names.split(",")

    def activation_tracker(module_name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            activations[module_name] = output
        return hook_fn

    activations = {}
    for module_name in module_names:
        module = model.get_submodule(module_name)
        module.register_forward_hook(activation_tracker(module_name))

    total_steps = 0
    original_size = image.height, image.width
    for resolution_exp in range(args.min_resolution, args.max_resolution):
        new_size = tuple(int(size * (args.resolution_scale ** resolution_exp)) for size in original_size)
        image = transforms.Resize(new_size)(image)

        image_tensor = torch.unsqueeze(to_tensor_transform(image), dim=0)
        image_tensor.requires_grad = True
        print(image_tensor.size())
        for step in range(args.num_steps):
            total_steps += 1
            loss = compute_loss(model=model, image=image_tensor, activations=activations, class_id=args.class_id)
            print(f"Step {total_steps}: Loss {loss.item()}")
            loss.backward()

            # This step to scale the image vector and clip it is very important to get this to work.
            grad = image_tensor.grad / (torch.std(image_tensor.grad) + 1e-8)
            image_tensor.data -= args.lr * grad
            image_tensor.data.clip_(min=-1, max=1)
            image_tensor.grad.data.zero_()

            with torch.no_grad():
                if total_steps > 0 and total_steps % args.display_every_n_steps == 0:
                    to_image_transform(transforms.Resize(original_size)(torch.squeeze(image_tensor, dim=0))).show()

        image = to_image_transform(torch.squeeze(image_tensor, dim=0))
        transforms.Resize(original_size)(image).show()


if __name__ == "__main__":
    main()
