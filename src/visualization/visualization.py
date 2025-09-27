# visualization/visualization.py
# 可视化模块：提供攻击过程的实时可视化和结果保存功能
# 这个模块使用matplotlib创建交互式窗口，显示原始图像、最优噪声和扰动结果
# 还包括保存各种分析图表的功能

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 使用交互式绘图后端
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 全局变量用于存储可视化窗口
fig: Optional[plt.Figure] = None
axs: Optional[np.ndarray] = None

def init_visualization(original_rgb: np.ndarray) -> None:
    """
    初始化可视化窗口。

    创建一个包含三个子图的窗口：
    - 左侧：原始图像
    - 中间：最优噪声热力图
    - 右侧：扰动后的图像

    Args:
        original_rgb: 原始RGB图像，形状[32, 32, 3]，值范围[0, 1]
    """
    global fig, axs
    plt.ion()  # 启用交互模式
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始图像
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # 预留中间图用于噪声显示
    axs[1].set_title("Best Noise")
    axs[1].axis("off")

    # 预留右侧图用于扰动图像显示
    axs[2].set_title("Perturbed Image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def update_visualization(
    original_rgb: np.ndarray,
    best_solution: Optional[Tuple[np.ndarray, np.ndarray]],
    best_obj: Tuple[bool, float, float, int],
    generation: int,
    mode: str,
    original_v: Optional[np.ndarray] = None
) -> None:
    """
    更新可视化窗口的内容。

    根据当前最优解更新噪声和扰动图像显示。

    Args:
        original_rgb: 原始RGB图像
        best_solution: 当前最优解 (indices, perturbations)，如果为None则跳过更新
        best_obj: 最优解的目标值 (is_adversarial, loss, l2_norm, l0_norm)
        generation: 当前代数
        mode: 扰动模式
        original_v: 原始V通道（仅v_channel模式）
    """
    if not fig or not best_solution:
        return

    is_adv, loss, l2, l0 = best_obj
    indices, perturbations = best_solution

    # 根据模式重建噪声和扰动图像
    if mode == "v_channel":
        # V通道模式：扰动只在亮度通道上
        noise_full = np.zeros_like(original_v, dtype=np.float32)
        noise_full.flat[indices] = perturbations
        original_hsv = rgb_to_hsv(original_rgb)
        perturbed_v = np.clip(original_hsv[:, :, 2] + noise_full, 0.0, 1.0)
        perturbed_hsv = np.stack([
            original_hsv[:, :, 0],
            original_hsv[:, :, 1],
            perturbed_v
        ], axis=-1)
        perturbed_rgb = hsv_to_rgb(perturbed_hsv)
        perturbed_rgb = np.clip(perturbed_rgb, 0.0, 1.0)

        # 噪声显示：归一化到[0,1]范围
        noise_display = noise_full.copy()
        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())
    else:
        # RGB模式：扰动在RGB空间
        noise_full = np.zeros_like(original_rgb, dtype=np.float32)
        noise_full.flat[indices] = perturbations
        perturbed_rgb = np.clip(original_rgb + noise_full, 0.0, 1.0)

        # 噪声显示：使用L2范数作为强度
        noise_display = np.linalg.norm(noise_full, axis=2)
        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())

    # 更新噪声热力图
    axs[1].clear()
    axs[1].imshow(noise_display, cmap="hot", interpolation="nearest")
    axs[1].set_title(f"Best Noise (Gen {generation})\nL2: {l2:.2f}, L0: {int(l0)}")
    axs[1].axis("off")

    # 更新扰动图像
    axs[2].clear()
    axs[2].imshow(perturbed_rgb)
    status = "SUCCESS" if is_adv else "FAIL"
    axs[2].set_title(f"Perturbed (Gen {generation})\nStatus: {status}, Loss: {loss:.3f}")
    axs[2].axis("off")

    plt.draw()
    plt.pause(0.01)  # 短暂暂停以更新显示

def save_noise_heatmap(noise_channel: np.ndarray, channel_name: str, save_path) -> None:
    """
    保存噪声热力图到文件。

    Args:
        noise_channel: 噪声数据数组
        channel_name: 通道名称（用于标题）
        save_path: 保存路径
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(noise_channel, cmap="hot", interpolation="nearest")
    plt.colorbar(im, shrink=0.8, label="Noise Magnitude")
    plt.title(f"Noise Heatmap - {channel_name} Channel")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_delta_L_heatmap(delta_L: np.ndarray, save_path) -> None:
    """
    保存ΔL*差值热力图（CIELAB颜色空间）。

    Args:
        delta_L: ΔL*差值数组，形状[32, 32]
        save_path: 保存路径
    """
    fig_deltaL, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(delta_L, cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
    plt.colorbar(im, shrink=0.8, label="ΔL* (Lab)")
    ax.set_title("ΔL* Difference (Perturbed - Original)")
    ax.axis("off")
    plt.tight_layout()
    fig_deltaL.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_deltaL)

def save_adversarial_result(
    original_rgb: np.ndarray,
    noise_display: np.ndarray,
    perturbed_rgb: np.ndarray,
    true_class_name: str,
    predicted_class_name: str,
    l0_norm: int,
    mode: str,
    save_path
) -> None:
    """
    保存最终对抗攻击结果图。

    创建包含原始图像、噪声和扰动图像的三联图。

    Args:
        original_rgb: 原始图像
        noise_display: 噪声显示数据（已归一化）
        perturbed_rgb: 扰动后的图像
        true_class_name: 原始类别名称
        predicted_class_name: 预测类别名称
        l0_norm: L0范数
        mode: 扰动模式
        save_path: 保存路径
    """
    fig_final, axs_final = plt.subplots(1, 3, figsize=(15, 5))

    axs_final[0].imshow(original_rgb)
    axs_final[0].set_title(f"Original ({true_class_name})")
    axs_final[0].axis("off")

    axs_final[1].imshow(noise_display, cmap="hot", interpolation="nearest")
    axs_final[1].set_title(f"Best Noise ({mode.upper()})\nL0: {l0_norm}")
    axs_final[1].axis("off")

    axs_final[2].imshow(perturbed_rgb)
    axs_final[2].set_title(f"Perturbed ({predicted_class_name})")
    axs_final[2].axis("off")

    plt.tight_layout()
    fig_final.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_final)

def save_convergence_plot(history_l2, history_l0, save_path) -> None:
    """
    保存收敛曲线图。

    显示L2和L0范数随代数的变化。

    Args:
        history_l2: L2范数历史记录
        history_l0: L0范数历史记录
        save_path: 保存路径
    """
    fig_conv, ax1 = plt.subplots(figsize=(10, 5))

    # 过滤出成功的代数（有对抗样本的代数）
    gens = np.arange(len(history_l2))
    successful_gens = [g for g, l2 in zip(gens, history_l2) if l2 is not None]
    successful_l2s = [l2 for l2 in history_l2 if l2 is not None]
    successful_l0s = [l0 for l0 in history_l0 if l0 is not None]

    if successful_l2s:
        color1 = "tab:red"
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("L2 Norm", color=color1)
        ax1.plot(successful_gens, successful_l2s, color=color1, label="L2 Norm")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("L0 Norm", color=color2)
        ax2.plot(successful_gens, successful_l0s, color=color2, label="L0 Norm")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(bottom=0)

        fig_conv.tight_layout()
        plt.title("Convergence of L2 and L0 Norms (Successful Attacks)")
        plt.savefig(save_path)
    plt.close(fig_conv)


def save_edge_guidance_artifacts(
    artifacts: Dict[str, np.ndarray],
    save_dir,
    prefix: str = "semantic_guidance",
) -> None:
    """Save diagnostic visuals for edge/semantic guidance artifacts as a single overview image."""

    def _normalize_for_display(array: np.ndarray) -> np.ndarray:
        data = np.asarray(array, dtype=np.float64)
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_min = float(np.min(data)) if data.size else 0.0
        data_max = float(np.max(data)) if data.size else 0.0
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
        return data

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    mapping = [
        ("gray", "Grayscale Reference", "gray"),
        ("edge_magnitude_base", "Base Edge Magnitude", "magma"),
        ("edge_magnitude_post_exponent", "Edges After Exponent", "inferno"),
        ("semantic_map", "Semantic Map", "viridis"),
        ("edge_magnitude_post_semantic", "Edges After Semantic Blend", "magma"),
        ("normalized_weights", "Normalized Sampling Weights", "plasma"),
    ]

    entries = []
    for key, title, cmap in mapping:
        if key in artifacts and isinstance(artifacts[key], np.ndarray) and artifacts[key].size > 0:
            entries.append((artifacts[key], title, cmap))

    if not entries:
        return

    num_entries = len(entries)
    cols = min(3, num_entries)
    cols = max(cols, 1)
    rows = (num_entries + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.atleast_2d(axes)
    elif cols == 1:
        axes = np.atleast_2d(axes).T

    for idx, (array, title, cmap) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        normalized = _normalize_for_display(array)
        im = ax.imshow(normalized, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Hide any unused axes
    for idx in range(num_entries, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    fig.tight_layout()
    output_path = target_dir / f"{prefix}_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)