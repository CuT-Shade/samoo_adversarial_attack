import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from pathlib import Path
from typing import Tuple, Optional, List
from matplotlib.colors import rgb_to_hsv
from torchvision.models import resnet18

def load_target_image_and_model(
    image_id: int, model_path: Path, data_root: Path, mode: str
) -> Tuple[np.ndarray, Optional[np.ndarray], int, nn.Module, List[str]]:
    transform = T.Compose([T.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    img_tensor, true_label = test_set[image_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    original_rgb_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    original_v_np = None
    if mode == "v_channel":
        original_hsv_np = rgb_to_hsv(original_rgb_np)
        original_v_np = original_hsv_np[:, :, 2]
    print(f"Loaded image id {image_id}, true label: {test_set.classes[true_label]} ({true_label})")
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    model.requires_grad_(False)
    return original_rgb_np, original_v_np, true_label, model, test_set.classes