import functools
from argparse import ArgumentParser
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import PIL
import PIL.Image
import lucent.optvis.param
import requests
import torch
import torchvision.models
from PIL.Image import Image
from lucent.optvis import transform
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import transforms

from consts import IMAGENET_MEAN
from consts import IMAGENET_STD
from deepdream.generate_from_noise import create_loss_fn

DEFAULT_STYLE_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_"
    "-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
)
DEFAULT_CONTENT_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Neckarfront_T%C3%BCbingen_Mai_2017.jpg/440px"
    "-Neckarfront_T%C3%BCbingen_Mai_2017.jpg"
)

def load_image(
    image_uri: Optional[str],
    height: int,
    width: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not image_uri:
        return None

    if image_uri.startswith("https"):
        r = requests.get(image_uri, headers={'User-Agent': 'My User Agent 1.0'}, stream=True)
        image = PIL.Image.open(r.raw)
    else:
        image = PIL.Image.open(image_uri)

    transform_image = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]
    )

    return torch.unsqueeze(transform_image(image).to(device), dim=0)


def track_activations(model: nn.Module) -> Dict[str, torch.Tensor]:
    activations = {}
    for name, module in model.named_modules():
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor, module_name: str) -> None:
            activations[module_name] = output

        module.register_forward_hook(functools.partial(hook_fn, module_name=name))

    return activations


def create_loss_fn(
    model: nn.Module,
    model_transform_fn: Callable[[torch.Tensor], torch.Tensor],
    activations: Dict[str, torch.Tensor],
    target_image: Optional[torch.Tensor],
    target_layers: List[str],
    feature_map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if target_image is None:
        return lambda x: torch.tensor(0, device=x.device)

    def loss_fn(image_tensor: torch.Tensor) -> torch.Tensor:
        model(model_transform_fn(torch.cat([image_tensor, target_image], dim=0)))

        losses = []
        for target_layer in target_layers:
            losses.append(F.l1_loss(
                feature_map_fn(activations[target_layer][0]),
                feature_map_fn(activations[target_layer][1]),
            ))

        return torch.sum(torch.stack(losses))

    return loss_fn


def convert_to_image(image_tensor: torch.Tensor) -> Image:
    return transforms.ToPILImage()(torch.squeeze(image_tensor, dim=0))


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_image", type=str, default=DEFAULT_CONTENT_IMAGE, help="Content image")
    parser.add_argument("--style_image", type=str, default=DEFAULT_STYLE_IMAGE, help="Style image")
    parser.add_argument(
        "--content_layers",
        type=str,
        default="features.10",
        help="A comma separated list of content layers",
    )
    parser.add_argument(
        "--style_layers",
        type=str,
        default="features.0,features.5,features.10,features.19,features.28",
        help="A comma separated list of style layers",
    )
    parser.add_argument("--steps", type=int, default=512, help="Number of optimization steps")
    parser.add_argument("--display_every_n_steps", type=int, default=100, help="Display progress every n steps")
    parser.add_argument("--init_stddev", type=float, default=1e-2, help="Initial image standard deviation")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--style_loss_weight", type=float, default=5.0, help="Style loss weight")
    parser.add_argument("--output_image", type=str, required=True, help="Output image")
    parser.add_argument("--output_height", type=int, default=512, help="Output image height")
    parser.add_argument("--output_width", type=int, default=512, help="Output image width")
    parser.add_argument(
        "--decorrelate", type=int, choices=[0, 1], default=1, help="Decorrelation trick for better style"
    )
    parser.add_argument(
        "--fft", type=int, choices=[0, 1], default=1, help="Fast Fourier Transform trick for better style"
    )
    args = parser.parse_args()

    assert args.content_image is not None or args.style_image is not None, (
        "At least one of content_image or style_image must be defined"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torchvision.models.vgg19(pretrained=True)
    model.to(device)
    model.eval()

    content_image = load_image(args.content_image, height=args.output_height, width=args.output_width, device=device)
    style_image = load_image(args.style_image, height=args.output_height, width=args.output_width, device=device)

    style_loss_weight = args.style_loss_weight if content_image is not None else 1.0

    model_transform_fn = transforms.Compose(
        transform.standard_transforms + [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)],
    )

    activations = track_activations(model)
    content_loss_fn = create_loss_fn(
        model=model,
        model_transform_fn=model_transform_fn,
        activations=activations,
        target_image=content_image,
        target_layers=args.content_layers.split(",") if args.content_layers else [],
        feature_map_fn=lambda x: x,
    )

    def gram_matrix(image: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape
        return torch.einsum("ik,jk->ij", image.view(c, -1), image.view(c, -1)) / (h * w)

    style_loss_fn = create_loss_fn(
        model=model,
        model_transform_fn=model_transform_fn,
        activations=activations,
        target_image=style_image,
        target_layers=args.content_layers.split(",") if args.content_layers else [],
        feature_map_fn=gram_matrix
    )

    _, _, h, w = content_image.size() if content_image is not None else style_image.size()
    # image_tensors[0] is already bound to tensor_transform.
    image_tensors, tensor_transform = lucent.optvis.param.image(
        w=w,
        h=h,
        sd=args.init_stddev,
        fft=args.fft,
        decorrelate=args.decorrelate,
    )

    optimizer = torch.optim.Adam(image_tensors, lr=args.lr)
    for step in range(args.steps):
        optimizer.zero_grad()
        content_loss = content_loss_fn(tensor_transform())
        style_loss = style_loss_fn(tensor_transform())
        total_loss = content_loss + style_loss_weight * style_loss

        total_loss.backward()
        optimizer.step()

        if step > 0 and (step % args.display_every_n_steps == 0 or step == args.steps - 1):
            print(
                f"Step {step}: Total loss {total_loss.item()}, "
                f"Content loss {content_loss.item()}, Style loss {style_loss.item()}"
            )
            convert_to_image(tensor_transform()).show()

    convert_to_image(tensor_transform()).save(args.output_image, format="png")


if __name__ == "__main__":
    main()
