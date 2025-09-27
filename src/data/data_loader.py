# data/data_loader.py
# 数据加载模块：负责加载目标图像、真实标签和预训练模型
# 这个模块包含了数据预处理、模型加载和初始化功能
# 支持不同的扰动模式，包括RGB和HSV颜色空间的处理

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
    """
    加载目标图像、真实标签与预训练模型。

    该函数从CIFAR-10测试集中加载指定索引的图像，转换为适合模型输入的格式，
    并加载预训练的ResNet18模型。同时根据扰动模式处理图像数据。

    Args:
        image_id: CIFAR-10测试集图像索引，取值范围[0, 9999]
        model_path: 预训练模型权重文件的路径
        data_root: CIFAR-10数据集的根目录路径
        mode: 扰动模式，决定是否额外返回V通道数据
             - "v_channel": 返回原始V通道数据
             - 其他模式: V通道返回None

    Returns:
        original_rgb: 原始RGB图像数组，形状为[32, 32, 3]，值范围[0, 1]
        original_v: 原始V通道数组（仅v_channel模式），形状为[32, 32]，值范围[0, 1]；
                   其他模式下为None
        true_label: 图像的真实类别索引（0-9）
        model: 加载好的ResNet18模型，已移动到适当设备并设为评估模式
        class_names: CIFAR-10的10个类别名称列表

    Raises:
        RuntimeError: 如果无法加载模型权重或数据集
    """
    # 定义图像预处理变换：转换为张量
    transform = T.Compose([T.ToTensor()])

    # 加载CIFAR-10测试集
    # train=False表示加载测试集，download=True会在需要时自动下载
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    # 获取指定索引的图像和标签
    img_tensor, true_label = test_set[image_id]

    # 确定计算设备：优先使用CUDA GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    # 将张量转换为NumPy数组，形状从[C, H, W]转换为[H, W, C]，值范围[0, 1]
    original_rgb_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # 根据扰动模式处理V通道数据
    original_v_np = None
    if mode == "v_channel":
        # 将RGB转换为HSV颜色空间
        original_hsv_np = rgb_to_hsv(original_rgb_np)
        # 提取V（亮度）通道
        original_v_np = original_hsv_np[:, :, 2]  # 形状[32, 32]

    # 打印加载信息
    print(f"Loaded image id {image_id}, true label: {test_set.classes[true_label]} ({true_label})")

    # 加载ResNet18模型
    # weights=None表示不使用预训练权重，我们将手动加载自定义权重
    model = resnet18(weights=None, num_classes=10)

    # 修改第一层卷积以适应32x32输入（CIFAR-10图像尺寸）
    # 原始ResNet18针对224x224图像，此处调整为3x3卷积，stride=1，padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 移除最大池化层，因为CIFAR-10图像已经很小
    model.maxpool = nn.Identity()

    # 加载自定义训练的权重
    # map_location=device确保权重加载到正确的设备
    # weights_only=True提高安全性，只加载权重参数
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    # 将模型移动到计算设备并设为评估模式（禁用dropout和batch norm更新）
    model.to(device).eval()

    # 禁用梯度计算，因为这是推理阶段
    model.requires_grad_(False)

    # 返回所有必要的数据
    return original_rgb_np, original_v_np, true_label, model, test_set.classes