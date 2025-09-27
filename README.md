# SA-MOO Adversarial Attack

这是一个精确复现论文 "Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation" (SA-MOO) 的Python实现。

## 功能特性

- **三种扰动模式**：
  - `rgb_sim`: 选中像素的RGB三通道同时修改
  - `channel`: 选中单个RGB通道修改（论文原始方式）
  - `v_channel`: 仅扰动HSV颜色空间的V（亮度）通道

- **两种扰动类型**：
  - **离散扰动**：扰动值从{-1, 0, 1}中选择（默认）
  - **连续扰动**：扰动值从指定范围内随机选择n位小数

- **攻击类型**：
  - 非目标攻击：让模型分类错误，但不指定具体类别
  - 目标攻击：让模型将图像分类为指定目标类别

- **多目标优化**：
  - 最小化L2范数（扰动幅度）
  - 最小化L0范数（扰动像素/通道数量）
  - 最大化对抗成功率

- **语义边缘引导**：通过 Sobel / Scharr / Prewitt / Canny 等算子生成边缘权重，可选多尺度融合与语义热力图加权，让初始化与变异更贴近关键结构。

## 项目结构

```
src/
├── config/
│   └── config.py          # 全局配置参数
├── utils/
│   ├── edge_guidance.py   # 语义边缘权重计算
│   └── logger.py          # 日志记录工具
├── data/
│   └── data_loader.py     # 数据加载和模型初始化
├── core/
│   ├── objectives.py      # 目标函数和支配关系
│   ├── evolutionary_operators.py  # 进化算子（初始化、交叉、变异、选择）
│   └── dynamic_parameters.py      # 动态超参数调度器
├── visualization/
│   └── visualization.py   # 可视化和结果保存
└── main.py                # 主执行文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

在 `src/config/config.py` 中修改以下关键参数：

- `TARGET_IMAGE_ID`: 目标图像在CIFAR-10测试集中的索引
- `MODEL_WEIGHTS_PATH`: 预训练ResNet18模型权重路径
- `PERTURBATION_MODE`: 扰动模式选择
- `IS_TARGETED_ATTACK`: 是否为目标攻击
- `FIXED_K`: 扰动预算（L0约束）
- `NUM_GENERATIONS`: 进化代数

### 连续扰动配置

- `ENABLE_CONTINUOUS_PERTURBATION`: 是否启用连续扰动（默认False）
- `CONTINUOUS_LOWER_BOUND`: 连续扰动下界（默认-1.0）
- `CONTINUOUS_UPPER_BOUND`: 连续扰动上界（默认1.0）
- `CONTINUOUS_DECIMAL_PLACES`: 小数位数（默认2）

启用连续扰动后，扰动值将从[下界,上界]范围内随机选择并四舍五入到指定小数位数。

### 动态支配关系配置

在 `config.yaml` 中的 `dominance` 段可以用来定义自适应的支配关系规则。每条规则包含：

- `when`: 触发条件，支持 `is_adversarial` 与 `other_is_adversarial` 等布尔组合。
- `prefer`: 直接指定更优个体（`self` / `other`）。
- `metrics`: 若省略 `prefer`，按顺序比较指定指标，支持 `goal: min|max` 与 `tolerance` 容差。
- `tie_breakers`: 全局兜底的比较指标，用于所有规则未决时的额外排序。

默认配置复现论文中“成功先于失败、L2优先”的支配关系，你可以通过增删规则或更改优先级，在不修改代码的情况下构建更复杂的多目标决策逻辑。

此外，可使用命令行或环境变量在线替换：

```bash
python run.py --dominance-config dominance_rules.json
# 或者
export SA_MOO_DOMINANCE_CONFIG='{"rules": [...]}'
```

### 动态参数调节

`dynamic_parameters` 段让关键超参数随代数演化自动调节。支持以下调度器：

- `constant`: 恒定值。
- `linear`: 线性插值，支持针对目标攻击和非目标攻击分别指定 `start`/`end`。
- `exponential`: 指数过渡（当 `start` 与 `end` 均大于0时生效）。
- `piecewise`: 分段常数调度，示例：

  ```yaml
  crossover_prob:
    schedule: piecewise
    pieces:
      - progress: 0.3
        value: 0.2
      - progress: 0.6
        value: 0.15
      - progress: 1.0
        value: 0.05
  ```

常见可调参数：

- `mutation_probability`
- `crossover_prob`
- `zero_sample_prob`
- `fixed_k`（实验性支持）

同样可通过 CLI / 环境变量注入新调度：

```bash
python run.py --dynamic-params dynamic_schedule.yaml
export SA_MOO_DYNAMIC_PARAMS='{"mutation_probability": {"schedule": "linear", "start": 0.3, "end": 0.05}}'
```

### 语义边缘引导

`edge_guidance` 段让初始化与变异更偏向图像语义轮廓：

- `enabled`: 是否启用边缘引导。禁用时回退到均匀采样。
- `method`: 边缘检测算子，支持 `sobel` / `scharr` / `prewitt` / `canny`。
- `gaussian_sigma`: 边缘检测前的高斯平滑强度。
- `exponent`: 对边缘强度的指数放大系数，增大可强化对强边缘的偏好。
- `uniform_mix`: 与均匀分布的混合比例，避免过度集中；取值范围 [0,1]。
- `min_value`: 归一化前添加的下限，防止概率为零。
- `canny_sigma`: 仅当 `method=canny` 时生效的边缘检测平滑系数。
- `multi_scale`: 多尺度配置集合：
  - `enabled`: 是否启用多尺度融合。
  - `scales`: 参与融合的尺度（缩放因子）列表，例如 `[1.0, 0.75, 0.5]`。
  - `sigma`: 对每个尺度追加的高斯平滑；可为单值或与 `scales` 等长的列表。
  - `aggregation`: 融合策略，支持 `max` / `mean` / `sum`。
- `semantic`: 语义图融合配置：
  - `enabled`: 是否加载语义分割图并参与融合。
  - `map_path`: 语义分割概率图或掩码的文件路径。
  - `weight`: 融合强度，取值越大语义引导越明显。
  - `normalize`: 是否对语义图归一化到 [0,1]。
  - `resize_mode`: 将语义图对齐到输入分辨率时的插值方式，支持 `bilinear` / `nearest`。
- `professional_preprocessing`: 使用专业模型输出进行预处理：
  - `enabled`: 是否启用专业模型的输出覆盖内部计算。
  - `edge_map_path`: 专业边缘检测输出文件路径（支持 png/jpg/npy/npz）。
  - `edge_normalize`: 是否对外部边缘图执行0-1归一化。
  - `semantic_map_path`: 专业语义分割输出文件路径。
  - `semantic_normalize`: 是否对外部语义图执行0-1归一化（随后仍可按 `semantic` 配置再处理）。
  - `fail_on_missing`: 是否在找不到外部文件时抛出异常（默认跳过）。
  - `semantic_model`: 直接在运行时调用专业语义分割模型（当前内置 `torchvision` 权重，例如 `deeplabv3_resnet50`）。
    - `name`: 模型名称，支持 `deeplabv3_resnet50` / `fcn_resnet50` / `lraspp_mobilenet_v3_large`。
    - `device`: `auto` / `cpu` / `cuda`，默认自动选择。
    - `output`: 输出类型（`max_prob`、`entropy`、`target_class`、`top2_gap`）。
  - `edge_model`: 使用专业模型或语义图生成高质量边缘图。
    - `type`: `semantic_gradient`（默认）、`semantic_entropy`、`torchvision_deeplabv3_resnet50`、`canny_rgb`。
    - `normalize` / `smooth_sigma`: 控制生成后的归一化与平滑。

配置示例：

```yaml
edge_guidance:
  enabled: true
  method: "sobel"
  gaussian_sigma: 0.8
  exponent: 1.5
  uniform_mix: 0.2
  min_value: 0.0001
  multi_scale:
    enabled: true
    scales: [1.0, 0.75, 0.5]
    aggregation: "max"
  semantic:
    enabled: true
    map_path: "./samples/img1049_semantic.png"
    weight: 0.6
  professional_preprocessing:
    enabled: true
    edge_map_path: "./external/hed_edges.png"
    edge_normalize: true
    semantic_map_path: "./external/deeplab_semantic.npy"
    semantic_normalize: true
    semantic_model:
      enabled: true
      name: "deeplabv3_resnet50"
      output: "max_prob"
    edge_model:
      enabled: true
      type: "semantic_gradient"
```

启用语义引导后，程序会自动在输出目录生成单张 `semantic_guidance_overview.png` 调试图（包含灰度参考、基础边缘、语义图、融合结果及最终采样权重），便于快速理解引导过程。

同样可以使用命令行或环境变量覆盖：

```bash
python run.py --edge-guidance edge_config.yaml
export SA_MOO_EDGE_GUIDANCE='{"enabled": true, "method": "canny", "canny_sigma": 1.2}'
```

## 环境变量

通过 `.env` 文件或系统环境变量设置：

- `MODEL_WEIGHTS_PATH`: 模型权重文件路径
- `DATA_ROOT_DIR`: CIFAR-10数据集路径
- `OUTPUT_DIR`: 输出结果保存目录
- `TORCH_HOME`: PyTorch缓存目录
- `LPIPS_CACHE_DIR`: LPIPS库缓存目录
- `SA_MOO_DOMINANCE_CONFIG`: 指向JSON/YAML文件或直接JSON字符串，用于覆盖支配关系
- `SA_MOO_DYNAMIC_PARAMS`: 指向JSON/YAML文件或直接JSON字符串，用于覆盖动态参数调度
- `SA_MOO_EDGE_GUIDANCE`: 指向JSON/YAML文件或直接JSON字符串，用于配置边缘引导参数

## 使用方法

### 🚀 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 使用默认配置运行
python run.py
```

### 🔧 使用配置文件运行实验

#### 方法1：使用预设配置文件
```bash
# RGB同步扰动实验
python run_experiment.py rgb_sim_config.yaml

# V通道（亮度）扰动实验
python run_experiment.py v_channel_config.yaml

# 目标攻击实验
python run_experiment.py targeted_attack_config.yaml

# 快速测试
python run_experiment.py quick_test_config.yaml
```

#### 方法2：使用命令行参数
```bash
# 直接指定参数（会覆盖默认配置）
python run.py --perturbation-mode v_channel --fixed-k 6 --num-generations 500

# 目标攻击
python run.py --targeted --target-class-id 3 --fixed-k 12

# 启用连续扰动
python run.py --enable-continuous-perturbation --continuous-lower-bound -0.5 --continuous-upper-bound 0.5 --continuous-decimal-places 3

# 指定自定义支配关系和动态参数调度
python run.py --dominance-config dominance_rules.yaml --dynamic-params dynamic_schedule.json
```
```

#### 方法3：使用环境变量
```bash
# 设置环境变量
export SA_MOO_PERTURBATION_MODE=v_channel
export SA_MOO_FIXED_K=6
python run.py
```

### 📁 配置文件管理

项目包含以下配置文件：

- **`complete_config.yaml`** - 📚 **完整配置参考**（包含所有参数和详细说明）
- **`config.yaml`** - ⚙️ 默认配置文件
- **`experiment_configs/`** - 🧪 实验专用配置文件夹

#### 使用完整配置

```bash
# 查看所有可用参数（完整配置）
python run_experiment.py complete_config.yaml

# 基于完整配置模板创建新实验
cp complete_config.yaml experiment_configs/my_experiment.yaml
# 编辑 my_experiment.yaml 中的参数
python run_experiment.py my_experiment.yaml
```

在 `experiment_configs/` 目录下可以创建多个配置文件：

```yaml
# 自定义配置文件示例
target_image_id: 1000
perturbation_mode: "channel"
fixed_k: 18
num_generations: 2000
is_targeted_attack: true
target_class_id: 7
```

然后使用：
```bash
python run_experiment.py your_custom_config.yaml
```

### 🔧 路径配置

**重要**：所有路径配置现在需要在 `.env` 文件中设置：

```bash
# .env 文件示例
MODEL_WEIGHTS_PATH=cifar10_resnet18.pth
DATA_ROOT_DIR=./data
OUTPUT_DIR=../attack_results
TORCH_HOME=/path/to/torch/cache
LPIPS_CACHE_DIR=/path/to/lpips/cache
```

路径配置的优先级：
1. **环境变量**（在 `.env` 文件中设置）
2. **命令行参数**（如 `--model-weights-path`）
3. **默认值**（如果环境变量和命令行参数都没有设置）

## 输出结果

程序会在 `OUTPUT_DIR` 下创建以时间戳命名的结果目录，包含：

- `report.txt`: 详细的执行日志
- `adversarial_result.png`: 对抗攻击结果对比图
- `noise_*.png`: 噪声热力图
- `deltaL_heatmap.png`: CIELAB颜色空间ΔL*差值图
- `convergence.png`: L2/L0范数收敛曲线
- `final_perturbed.png`: 32x32扰动图像（可直接用于测试）
- `semantic_guidance_overview.png`: 若启用语义引导，会额外保存包含灰度、基础边缘、语义图及融合权重的综合诊断图

## 评估指标

- **L2范数**: 扰动的欧几里得范数
- **L0范数**: 非零扰动的像素/通道数量
- **PSNR**: 峰值信噪比
- **LPIPS**: 感知损失
- **ΔE00**: CIELAB颜色空间色差

## 注意事项

1. 确保模型权重文件存在且路径正确
2. 对于`v_channel`模式，建议设置`FIXED_K=6`
3. 目标攻击需要设置`TARGET_CLASS_ID`
4. 可视化功能需要图形界面支持

## 引用

如果使用此代码，请引用原始论文：

```
Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation
```

## MIT许可证

本项目遵循MIT许可证。
