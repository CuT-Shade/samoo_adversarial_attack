# SA-MOO 稀疏对抗攻击框架 & RobustBench 评测套件

_Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation_ 复现 + 扩展实现  
支持：进化多目标优化 / 语义边缘引导 / 频域低频约束 / 动态支配关系 / 参数批量搜索 / Optuna 多目标调参 / RobustBench 离线评测。

---

## 🔥 核心特性一览

| 类别 | 能力 | 说明 |
| ---- | ---- | ---- |
| 扰动模式 | `rgb_sim` / `channel` / `v_channel` | 同步 RGB / 单通道 / HSV 亮度通道 |
| 扰动取值 | 离散 {-1,0,1} / 连续区间 | 连续模式支持上下界与小数精度控制 |
| 目标类型 | Untargeted / Targeted | 支持指定 `target_class_id` |
| 优化目标 | L0 / L2 / 成功率 / (附加观测 LPIPS, ΔE00) | 多目标 + 可配置支配规则 |
| 语义引导 | 边缘 / 多尺度 / 语义图 / 专业模型融合 | 自动生成诊断可视化 |
| 频域约束 | DCT 低频滤波 + 边缘能量再分配 | 仅前 N 代或全程启用 |
| 动态调度 | 交叉 / 变异 / K / 采样概率等 | 线性 / 指数 / 分段 |
| 约束防护 | 查询次数 / L0 上限 | 失败 Trial 统计 & 剔除 |
| 批量搜索 | 参数 Sweep + 结果收集 | 自动生成 slug & summary 表 |
| 多目标调参 | Optuna 四目标 (成功率, L0, LPIPS, Queries) | 帕累托前沿导出 + 图像库 |
| 离线评测 | RobustBench 模型 + 数据集缓存 | 自动复制权重到相对路径，`--no-download` |
| 可视化 | 收敛曲线 / 三联图 / 频域热力图 / 语义融合图 | 支持 harvest 精简收集 |
| UI | Streamlit 面板 | 交互筛选 + 参数相关性分析 |

---

## 📑 目录 (Table of Contents)

1. [快速开始](#-快速开始)
2. [安装与环境](#-安装与环境)
3. [.env 配置](#-env-配置)
4. [目录结构概览](#-目录结构概览)
5. [单次攻击与基础脚本](#-单次攻击与基础脚本)
6. [RobustBench 评测流程](#-robustbench-评测流程)
7. [批量多模式评测与统计](#-批量多模式评测与统计)
8. [核心参数与攻击配置](#-核心参数与攻击配置)
9. [高级模块：语义边缘 / DCT 低频 / 动态支配 / 动态参数](#-高级模块)
10. [批量参数 Sweep](#-批量参数-sweep)
11. [Optuna 多目标搜索](#-optuna-多目标搜索)
12. [Streamlit 可视化](#-streamlit-可视化)
13. [输出文件说明](#-输出文件说明)
14. [常见问题 FAQ & 故障排查](#-faq--故障排查)
15. [复现与性能建议](#-复现与性能建议)
16. [引用](#-引用)
17. [许可证](#-许可证)

## 🚀 快速开始

```powershell
conda create -n Lab_Round python=3.12
conda activate Lab_Round
pip install -r requirements.txt
```

最小一次攻击（使用默认 config）：
```powershell
python run.py
```

RobustBench 单模型评测（离线 + 调试路径）：
```powershell
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 20 --no-download --debug-paths
```

参数帮助：
```powershell
python -m src.robustbench_eval --help
python run.py --help
```

---

## 🧱 安装与环境

附加依赖：
```powershell
pip install robustbench
pip install git+https://github.com/fra31/auto-attack
pip install streamlit optuna
```

可选加速/可视化：`tensorboard`, `seaborn`, `pandas`。

---

## ⚙️ .env 配置

根目录新建 `.env`：
```ini
MODEL_WEIGHTS_PATH=E:/path/to/custom_model.pth
DATA_ROOT_DIR=E:/datasets/cifar-10-batches-py
OUTPUT_DIR=./attack_results
TORCH_HOME=E:/cache/torch
LPIPS_CACHE_DIR=E:/cache/torch
ROBUST_BENCH_DIR=E:/cache/torch
SA_MOO_SEED=1234
```
说明：
- `ROBUST_BENCH_DIR`：放置标准模型权重，脚本会自动复制到 `./models/...` 兼容相对加载。
- `DATA_ROOT_DIR`：若存在则跳过 CIFAR 自动下载。
- 可通过命令行覆盖 `.env`。

---

## 🗂 目录结构概览
```
src/
  core/ (攻击主逻辑)
  config/ (全局/覆盖配置)
  utils/ (边缘, DCT, 日志等)
  visualization/
  robustbench_eval.py (标准模型评测)
run.py / run_experiment.py / run_parameter_sweep.py / ultimate_optimizer.py
streamlit_app.py
experiment_configs/  sweep_configs/  optuna_results/  sweep_runs/
attack_results/  model_cache/  data/
```

---

## 🎯 单次攻击与基础脚本

| 场景 | 命令 | 说明 |
| ---- | ---- | ---- |
| 默认攻击 | `python run.py` | 使用 `config.yaml`/环境变量 |
| 指定配置 | `python run_experiment.py experiment_configs/xxx.yaml` | 读取完整 YAML |
| 改写参数 | `python run.py --perturbation-mode v_channel --fixed-k 8` | 覆盖局部参数 |
| 目标攻击 | `python run.py --targeted --target-class-id 3` | 设置目标类别 |
| 连续扰动 | `python run.py --enable-continuous-perturbation --continuous-upper-bound 0.6` | 启用连续值 |

---

## 🧪 RobustBench 评测流程

### 1. 放置 / 准备权重
```
ROBUST_BENCH_DIR/models/cifar10/Linf/Standard.pt
```
若已存在其他层级（如 `robustbench/robustbench/models/...`），脚本会自动探测。

### 2. 运行评测
```powershell
# 基础
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 100

# 离线严格模式（不存在即报错）
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 50 --no-download

# 指定起始索引 & 保存每样本 artefact
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --start-index 500 --n-examples 32 --save-artifacts

# 调试路径
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 1 --debug-paths
```

也可以直接使用config.yaml哦：
```powershell
python -m src.robustbench_batch_modes --config config/batch_modes_config.yaml
```
或者
```powershell
python -m src.robustbench_eval `
  --model-name Standard `
  --dataset cifar10 `
  --threat-model Linf `
  --config experiment_configs/robustbench_v_channel_random.yaml `
  --n-examples 100 `
  --random-sample `
  --random-seed 42 `
  --output-dir rb_runs/v_channel_seed42 `
  --results-json rb_runs/v_channel_seed42/results.json `
  --no-download `
  --verbose
```

### 3. 结果指标
输出 JSON 含：`robust_accuracy` (= 1 - 成功率 for untargeted)、`average_queries`、成功数等。

### 4. 常见问题
| 现象 | 解决 |
| ---- | ---- |
| 一直打印 Downloading | 已被 monkey patch；若仍网络失败，确保本地权重路径正确 |
| FileNotFoundError: models/... | 未复制到相对路径；重新运行带 `--debug-paths` 自动复制 |
| 数据集再次下载 | 未设置 `DATA_ROOT_DIR` 或路径层级不匹配 (需直接指向包含 `data_batch_1` 的文件夹) |

---

## 🧩 批量多模式评测与统计

一次性对同一批样本（共享随机或自定义索引）运行三种扰动模式，并输出对比表与聚合指标。

### 1. 快速示例

```powershell
python -m src.robustbench_batch_modes `
  --model-name Standard `
  --dataset cifar10 `
  --threat-model Linf `
  --n-examples 100 `
  --random-sample `
  --random-seed 42 `
  --modes rgb_sim,channel,v_channel `
  --base-output-dir batch_runs_cifar10 `
  --no-download `
  --verbose
```

### 2. 参数要点
| 参数 | 作用 |
| ---- | ---- |
| `--indices` | 显式样本索引（最高优先级） |
| `--indices-file` | 从 json/txt 读取索引；自动解析逗号或换行分隔 |
| `--random-sample` + `--random-seed` | 随机采样并持久化 `indices.json`，确保多次运行复现 |
| `--modes` | 逗号分隔的扰动模式列表（默认三种） |
| `--base-output-dir` | 所有模式输出目录的根路径 |
| `--save-artifacts` | 保存每个样本的对抗结果、日志 |
| 其他参数 | 原评测脚本的所有 CLI 参数均可透传 |

### 3. 输出结构
```
batch_runs_cifar10/
  indices.json
  rgb_sim/
    results.json
    run_stdout.log
    sample_00012/ ... (启用 --save-artifacts 时生成)
  channel/
    results.json
    run_stdout.log
  v_channel/
    results.json
    run_stdout.log
  aggregate.json
  aggregate.csv
```

`aggregate.csv` 会包含关键列：`mode`, `n_examples`, `success_count`, `robust_accuracy`, `average_queries`，以及记录在 `metrics` 中的数值指标均值（成功样本上计算，命名为 `mean_*`）。

### 4. 复现与扩展建议
- 首次运行可用 `--random-sample --random-seed` 生成索引，后续通过 `--indices-file indices.json` 固定样本。
- 如需并行多模式、绘制对比图或自定义聚合策略，可在当前脚本基础上扩展；欢迎提出需求。

---

## 🔧 核心参数与攻击配置

| 参数 | 作用 | 备注 |
| ---- | ---- | ---- |
| `fixed_k` | L0 像素/通道预算 | 稀疏度控制 |
| `num_generations` | 进化代数 | 更多代=潜在更优但耗时更高 |
| `population_size` | 种群规模 | 大规模提升多样性 |
| `perturbation_mode` | 扰动模式 | `rgb_sim`/`channel`/`v_channel` |
| `enable_continuous_perturbation` | 启用连续值 | 结合上下界与精度 |
| `is_targeted_attack` & `target_class_id` | 目标攻击 | 需提供类索引 |
| `zero_sample_prob` | 采样保留原像素概率 | 稀疏初始化 |
| `crossover_prob` | 交叉概率 | 影响探索/开发平衡 |
| `mutation_probability` | 变异概率 | 随代动态可调 |

更多高级配置见 `complete_config.yaml` 与下面模块说明。

---

## 🧬 高级模块

### 语义边缘引导
多尺度边缘 + 语义图 + 专业分割模型结果融合，生成采样概率热图；输出 `semantic_guidance_overview.png`。

### DCT 低频约束
前 N 代限制扰动主要在低频，减少早期搜索噪声；产生 `freq_low_heatmap.png` / `freq_high_heatmap.png`。

### 动态支配关系 (dominance)
按规则决定非支配比较优先级（成功优先 / L2 优先 / 带容差）；可 JSON/YAML 注入。

### 动态参数调度 (dynamic_parameters)
支持 linear / exponential / piecewise；适用于交叉、变异、K 等。可通过环境变量热更新。

---

## 🔁 批量参数 Sweep

配置模板：`sweep_configs/*.yaml`。
```powershell
python run_parameter_sweep.py sweep_configs/my_first_sweep.yaml --max-runs 50
```
生成：`sweep_runs/<timestamp>/summary.(csv|json)` + 精选 artefacts。

---

## 🎯 Optuna 多目标搜索

四目标：(maximize) Success, (minimize) L0, LPIPS, Queries。
```powershell
python ultimate_optimizer.py --base-config complete_config.yaml --study-name demo --n-trials 40 --repeats 3
```
结果：`optuna_results/<study>/` 帕累托前沿 + 图像库。

---

## 📊 Streamlit 可视化
```powershell
streamlit run streamlit_app.py
```
筛选 + 散点 / 箱线图 / 并行坐标 / 相关性热图 / 运行详情。

---

## 📁 输出文件说明

| 文件 | 含义 |
| ---- | ---- |
| `report.txt` | 文本日志与关键指标 |
| `adversarial_result.png` | 原图 / 扰动 / 对抗图三联 |
| `final_perturbed.png` | 仅对抗结果 (32x32) |
| `noise_*.png` | 噪声分布热力图 |
| `convergence.png` | L0 / L2 演化趋势 |
| `semantic_guidance_overview.png` | 语义融合调试图 |
| `freq_low/high_heatmap.png` | DCT 频域能量分布 |
| `metrics.json` | 结构化指标（若配置输出） |
| `summary.json` | RobustBench 评测汇总 |

---

## ❓ FAQ & 故障排查

| 问题 | 可能原因 | 解决 |
| ---- | -------- | ---- |
| 模型仍显示 Downloading | 版本日志习惯 | 查看是否真正发起网络；`--no-download` + 确认权重路径 |
| FileNotFoundError: models/... | 未复制相对路径 | 重跑带 `--debug-paths` 或手动复制 |
| 数据集重复下载 | 指向错误上层目录 | `DATA_ROOT_DIR` 需含 `data_batch_1` 文件 |
| LPIPS 权重缺失 | 未设置缓存 | 确认 `LPIPS_CACHE_DIR` 或保持联网一次 |
| 目标攻击无效 | 未传 `--target-class-id` | 补充参数或检查类索引 |
| L0 超出预期 | 连续模式+小数未被稀疏优化 | 调整 `fixed_k` 或增大 `zero_sample_prob` |
| 多目标调参过慢 | `population_size` 太大 | 减小种群或降低 `num_generations` |

---

## 🧪 复现与性能建议
- 固定 `SA_MOO_SEED`，同时设定 torch / numpy / random。
- 首轮可用：`population_size=20, num_generations=400, fixed_k=6` 验证流程。
- 频域 + 语义引导同时启用时，可降低初始变异概率加快收敛。
- Sweep 前先缩短代数验证指令正确性，再放大代数。

---

## 📚 引用
```
Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation
```

---

## 📄 许可证
仅供学术研究使用。请在成果中注明来源。

---

如需英文版 README 或补充脚本示例，请提交 Issue/联系维护者。

## 项目简介
本项目实现了 SA-MOO 多目标进化攻击算法，并集成了 RobustBench 标准模型评测接口。支持离线缓存模型权重和数据集，支持自动读取 `.env` 配置，适合大规模实验和复现。

---

## 环境准备

建议使用 Anaconda 创建独立环境：

```powershell
conda create -n Lab_Round python=3.12
conda activate Lab_Round
pip install -r requirements.txt
```

如需 RobustBench 及 AutoAttack：
```powershell
pip install robustbench
pip install git+https://github.com/fra31/auto-attack
```

---

## .env 配置说明

项目根目录下的 `.env` 文件可配置所有关键路径：

```ini
MODEL_WEIGHTS_PATH=... # 你的自定义模型权重（非 RobustBench 标准模型可选）
DATA_ROOT_DIR=E:/.../data/cifar-10-batches-py # 数据集主目录，含 data_batch_1 等文件
OUTPUT_DIR=./attack_results # 攻击结果输出目录
TORCH_HOME=E:/.../model_cache # PyTorch/RobustBench 权重缓存主目录
LPIPS_CACHE_DIR=E:/.../model_cache # LPIPS 感知损失权重缓存
ROBUST_BENCH_DIR=E:/.../model_cache # RobustBench 权重缓存主目录
```

- 所有路径均可用绝对或相对路径。
- `.env` 会被 `robustbench_eval.py` 自动读取，无需手动传参。

---

## 权重与数据集缓存

- **模型权重**：
  - RobustBench 标准模型权重需放在 `ROBUST_BENCH_DIR/models/<dataset>/<threat_model>/<model_name>.pt`。
  - 运行时会自动复制到当前工作目录下的 `models/...`，无需手动操作。
  - 非标准模型可用 `MODEL_WEIGHTS_PATH` 指定。

- **数据集**：
  - CIFAR-10/100/Imagenet 数据集需放在 `DATA_ROOT_DIR` 下。
  - 目录结构需包含 `cifar-10-batches-py` 或 `cifar-100-python` 文件夹。
  - 若本地无数据集，RobustBench 会自动下载。

---

## 评测脚本用法

主入口：`src/robustbench_eval.py`

### 基本评测

```powershell
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 100 --output-dir rb_test
```

### 强制只用本地缓存（不下载）

```powershell
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 100 --output-dir rb_test --no-download
```

### 调试路径（显示所有关键路径）

```powershell
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --n-examples 1 --output-dir rb_test --debug-paths
```

### 指定自定义数据集路径

```powershell
python -m src.robustbench_eval --model-name Standard --dataset cifar10 --data-dir "E:/.../data/cifar-10-batches-py" --output-dir rb_test
```

---

## 高级功能

- 支持所有 SA-MOO 参数自定义（如种群规模、迭代次数、攻击模式等），详见 `--help`。
- 支持 targeted/untargeted 攻击。
- 支持自动保存每个样本的攻击结果与日志。
- 支持多种输出格式（JSON、图片、日志等）。
- 支持 monkey patch，彻底阻断 RobustBench 权重下载。

---

## 常见问题 FAQ

**Q: 为什么模型权重已存在还会下载？**
A: RobustBench 某些版本只认当前工作目录下的 `models/...`，本项目已自动复制权重，无需手动干预。

**Q: 数据集下载很慢怎么办？**
A: 建议提前下载好数据集并配置 `.env` 的 `DATA_ROOT_DIR`，或用 `--data-dir` 参数指定。

**Q: 如何批量评测多个模型？**
A: 可用 shell 脚本或 Python 循环批量调用 `robustbench_eval.py`，每次指定不同 `--model-name`。

**Q: 如何自定义攻击参数？**
A: 运行 `python -m src.robustbench_eval --help` 查看所有可选参数。

---

## 结果输出说明

- 评测 summary 会在终端和 JSON 文件中输出，包括：
  - `model_name`、`dataset`、`threat_model`、`n_examples`、`success_count`、`robust_accuracy`、`average_queries`
- 每个样本的详细攻击结果会保存到 `output_dir` 下。

---

## 维护与扩展

如需集成新模型、新攻击方法或自定义评测流程，请参考 `src/core/attack_runner.py` 和 `src/main.py`，或联系项目维护者。

---

## 致谢

本项目参考并集成了 [RobustBench](https://github.com/RobustBench/robustbench)、[AutoAttack](https://github.com/fra31/auto-attack) 等开源工具。

---

如有任何问题或建议，欢迎 issue 或私信反馈！
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
- **频域约束模块**：新增可开关的 DCT 低频干扰模块，可在前 N 代强制扰动落在低频区域，并支持与语义边缘引导联动以优先保留关键轮廓。

## 项目结构

```
src/
├── config/
│   └── config.py          # 全局配置参数
├── utils/
│   ├── edge_guidance.py   # 语义边缘权重计算
│   ├── dct_low_frequency.py  # DCT低频干扰模块
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

## 批量参数评测

为了批量探索不同的超参数组合，可使用新增的 `run_parameter_sweep.py` 工具。它会根据 YAML 规格展开参数网格，自动生成临时配置并串行/并行调用 `run_experiment.py`，最终汇总每次实验的核心指标。

**快速上手：**

1. 复制 `sweep_configs/sweep_template.yaml` 为新的规格文件（例如 `sweep_configs/my_first_sweep.yaml`），按注释修改：
  - `base_config`：继承的基础配置；可以直接使用 `config_template.yaml`。
  - `overrides`：本轮 sweep 的统一固定项（如 `target_image_id`、`num_generations`）。
  - `parameters`：使用点语法声明需要搜索的参数及其候选值列表。
  - `ranking.priority`：定义排序优先级及方向，便于快速筛选“最佳”组合。
2. 在项目根目录运行：
  ```bash
  python run_parameter_sweep.py sweep_configs/my_first_sweep.yaml
  ```
  可附加 `--max-runs` 限制组合数量，或使用 `--max-random-runs` 随机抽样指定数量的组合（配合 `--random-seed` 可复现），也可以先加 `--dry-run` 查看展开结果。
3. 脚本会生成两个输出位置：
  - `attack_results/sweep_<时间戳>/`：每个组合对应一个标准攻击输出文件夹。
  - `sweep_runs/sweep_<时间戳>/`：汇总工作区，包含
    - `summary.csv` / `summary.json`：全部组合的指标表
    - `stdout/`、`stderr/`：每个实验的原始终端日志
    - `configs/`：自动生成的配置文件（若 `keep_configs` 为 true）
    - `harvest/`：按需拷贝的诊断图（来源于 `harvest` 字段，可例如收集频谱热力图、对抗样本）

### 结果分析与追踪

- **概览排序**：`summary.csv` 已按排名规则排序，可直接在表格工具或 Pandas 中筛选：

  ```python
  import pandas as pd

  df = pd.read_csv("sweep_runs/sweep_20250928_204033/summary.csv")
  top = df.query("status == 'ok' and success == 1").nsmallest(5, "l2")
  print(top[["index", "slug", "l2", "confidence_drop", "runtime_sec"]])
  ```

  `status` 字段标识运行是否成功解析结果，`runtime_sec` 有助于定位耗时配置。

- **逐项复盘**：摘要中的 `slug` 与 `index` 对应 `attack_results/sweep_*` 下的运行目录，可在 `stdout/<index>_*.log` 或 `stderr/` 中查看完整日志。

- **对抗样本与报告**：默认 `harvest/` 会收集 `report.txt`、`adversarial_result.png`（三联图）、`final_perturbed.png`（32×32 对抗图像）等文件，便于快速比较不同组合的可视化效果。

- **批量指标对比**：利用 `summary.json` 可在脚本中绘制趋势，例如统计不同 `population_size` 的成功率 / 平均 L2：

  ```python
  import json
  from collections import defaultdict

  with open("sweep_runs/sweep_20250928_204033/summary.json", "r", encoding="utf-8") as fh:
      records = json.load(fh)

  grouped = defaultdict(list)
  for rec in records:
      slug = rec["slug"]
      pop = slug.split("__population_size=")[1].split("__", 1)[0]
      if rec.get("success") == 1:
          grouped[pop].append(rec["l2"])

  for pop, values in grouped.items():
      print(pop, sum(values) / len(values))
  ```

### Streamlit 可视化面板

想要在浏览器里直接探索 sweep 结果，可以使用根目录下的 `streamlit_app.py`：

```powershell
streamlit run streamlit_app.py
```

运行后可在侧边栏选择具体的 `sweep_runs/<时间戳>`，随后即可：

- 通过多选框 / 滑块过滤运行状态、成功标记、数值指标范围以及关键超参数（扰动模式、是否目标攻击、连续扰动模式等）；
- 在“Table”页查看按指标排序后的 TOP 配置，支持控制排序字段与显示的条目数量；
- 在“Visualizations”页绘制自定义散点图（例如 `confidence_drop` vs `l2`），并按参数类别着色，同时按任意参数分组统计成功率、平均指标等；
- 在“Parameter insights”页利用箱线图、并行分类、并行坐标和相关性热图，快速观察参数取值与关键指标之间的对应关系；
- 在“Run details”页查看指定组合的完整 slug、输出目录、生成的配置文件（若 `keep_configs=true`）以及 `harvest` 目录中的对抗图像、收敛曲线等。

发生过滤后，图表和表格会实时更新，帮助你快速定位最优或最稳健的参数组合。

- `value` / `overrides` 扩展：当某个取值需要额外绑定其它字段时，可将列表项写成
  
  ```yaml
  is_targeted_attack:
    - value: false
    - value: true
      overrides:
        target_class_id: 5
  ```

  脚本会先把主键设置为 `value`，随后依次应用 `overrides` 中的额外字段。`overrides` 支持任意点语法键，可用于成对修改上下界、同步开关等依赖关系。

汇总表中默认解析的指标包括：

- `success`（对抗是否成功，1/0）
- `l2` / `l0` / `psnr`
- `lpips`
- `mean_delta_e00` / `max_delta_e00`
- `orig_confidence` / `adv_confidence` / `confidence_drop`
- `runtime_sec`

如果定义了 `ranking.priority`，脚本还会计算归一化综合得分（`composite_score`）并按优先级排序。这样可以快速定位“成功率最高、L2 最小”的组合，或筛选感知质量最好的扰动。

## Optuna 多目标超参搜索

针对同时优化成功率、扰动 sparsity 与感知质量的复杂场景，可以使用根目录新增的 `ultimate_optimizer.py`。该脚本基于 Optuna 提供多目标研究工作流，具备以下特性：

- **四目标联合优化**：最大化对抗成功率，同时最小化 L0、LPIPS、查询次数（L2 仍会记录用于分析但不参与 Optuna 目标）。
- **丰富搜索空间**：自动采样种群大小、最大代数、固定 K、交叉/零采样概率、连续扰动开关与范围、语义边缘引导及其语义引导参数。
- **持久化与恢复**：默认使用 `sqlite:///samoo-attack-opt.db`，多次执行同一 `--study-name` 即可从断点续跑；按 `Ctrl+C` 中断时会自动生成当前帕累托前沿报告。
- **重复评估估计成功率**：通过 `--repeats` 指定每个 Trial 的独立重复次数，用统一种子实现可复现的蒙特卡洛估计。
- **硬约束守护**：新增 `--max-queries` 与 `--max-l0` 参数（默认分别为 10000 与 30，可按需覆盖），任何重复若超出查询预算或 L0 稀疏度，即被标记为失败并计入违反次数，确保搜索结果满足业务约束。
- **即刻产出可视化与图像库**：每次生成/中断后都会更新 `optuna_results/<study_name>/` 下的 CSV、JSON、HTML/PNG 帕累托前沿，以及从帕累托 Trial 拷贝的 `adversarial_result.png`/`final_perturbed.png` 等对比图；若缺少 Chrome/Edge 浏览器内核，PNG 导出会自动跳过并在终端提示。

### 使用示例

```powershell
python ultimate_optimizer.py \
  --base-config complete_config.yaml \
  --study-name samoo_rgb_sim \
  --perturbation-mode rgb_sim \
  --repeats 3 \
  --n-trials 40 \
  --max-queries 10000 \
  --max-l0 30
```

- 如果省略 `--max-queries` / `--max-l0`，脚本将分别采用 10000 次查询与 30 像素/通道的默认上限。
- 通过 `--output-dir` 可以将所有 Trial 的攻击结果集中保存到指定目录；
- `--seed` 设置全局伪随机种子，使每次重复具备可复现性；
- `--cleanup-runs` 会在采集指标后删除 `attack_results/...` 产生的原始目录，只保留 Optuna 工作区内的聚合信息与关键图像；
- `--n-jobs` 支持多进程 Trial 并行（每个 Trial 内的重复仍串行执行，以复用 GPU/显存）。
- `trial_metrics.csv/json` 新增 `constraint_violations`、`raw_average_l0`、`raw_average_queries` 与 `constraints_ok` 字段，可快速判断原始均值与是否存在超限重复（`constraint_violations>0` 表示至少有一次重复超限）。

### 结果目录结构

```
optuna_results/<study_name>/
├── trial_metrics.csv        # 全部 Trial 指标总览
├── trial_metrics.json       # 同上（JSON）
├── pareto_front.json        # 当前帕累托集
├── pareto_front_2d.html/png # 查询次数 vs LPIPS 散点
├── pareto_front_3d.html/png # L0 vs 查询次数 vs LPIPS
├── pareto_solutions/        # 帕累托解的对抗三联图、扰动热力图
└── trials/
  └── trial_<id>/          # 单个 Trial 的配置、日志与重复结果
    ├── overrides.yaml
    ├── aggregate.json
    └── repeat_XX/
      ├── config.yaml
      ├── metrics.json
      ├── stdout.log / stderr.log
      └── (链接到 attack_results/... 中的原始 artefact)
```

每个 Trial 完成后，`metrics.json` 会保存到运行目录及 `repeat_XX/` 下，方便独立分析或自定义可视化。

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

### DCT 低频干扰模块

`dct_low_frequency` 段提供了一个可开关的频域约束模块，会在RGB空间重建之后、模型前向之前，将扰动投影到DCT低频系数中。可通过 `max_generations` 控制仅在前 N 代生效，后续迭代将恢复原始稀疏表示。启用后：

- 在受限代数内，对模型评估时的候选样本自动执行低频滤波，使早期搜索集中于大尺度扰动；
- 可配合 `v_channel`/`rgb_sim`/`channel` 三种模式使用，无需修改进化算子；
- 若语义边缘引导已启用，将在低频滤波后按边缘概率图重新分配能量，优先在显著轮廓附近保留扰动。
- 生成最终报告时，会额外导出低/高频热力图 `freq_low_heatmap.png` 与 `freq_high_heatmap.png`，直观展示频域分解后保留与剔除能量的空间分布。

关键参数：

- `enabled`: 是否启用低频干扰。
- `max_generations`: 低频约束持续的迭代数；<=0 表示全程启用。
- `keep_ratio`: 沿行/列保留的低频比例，0-1 之间。
- `keep_rows` / `keep_cols`: （可选）显式指定保留的系数数目，优先级高于 `keep_ratio`。
- `min_rows` / `min_cols`: 至少保留的行/列数。
- `blend`: 0 表示完全保留原扰动，1 表示完全替换为低频重建，可填写介于 0-1 的混合比例。
- `boost`: 对保留系数施加的放大因子（>1 放大低频，<1 衰减）。
- `decay`: 可选的余弦平滑指数，用于软化低频掩码边缘。
- `channel_weights`: 长度为 3 的列表，为 RGB 三通道分别设置低频混合权重。
- `epsilon`: 扰动能量阈值，小于该阈值时跳过低频操作以避免数值噪声。
- `edge_blend`: 与语义边缘权重的融合强度，1 表示完全依赖边缘分布，0 表示忽略边缘。
- `edge_exponent`: 边缘权重指数放大系数，可增强对高响应轮廓的偏好。
- `preserve_energy`: 是否在应用边缘裁剪后重新缩放扰动能量。

示例配置：

```yaml
dct_low_frequency:
  enabled: true
  max_generations: 50
  keep_ratio: 0.3
  blend: 0.85
  boost: 1.1
  min_rows: 2
  min_cols: 2
  decay: 1.5
  channel_weights: [1.0, 0.9, 0.9]
  edge_blend: 0.8
  edge_exponent: 1.5
  preserve_energy: true
```

命令行与环境变量也支持覆盖：

```bash
python run.py --dct-low-frequency dct_config.yaml
export SA_MOO_DCT_LOW_FREQ='{"enabled": true, "keep_ratio": 0.2, "blend": 0.75}'
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
- `SA_MOO_DCT_LOW_FREQ`: 指向JSON/YAML文件或直接JSON字符串，用于配置DCT低频干扰参数
- `SA_MOO_SEED`: 指定整数种子，同步设置 NumPy、Python `random` 与 PyTorch 随机源，以便重复实验获得一致结果
- `SA_MOO_METRICS_JSON`: 当设置为文件路径时，`run.py`/`run_experiment.py` 会在额外位置写出本次攻击的 `metrics.json`

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
- `freq_low_heatmap.png`: 高斯低频分量的热力图，可视化扰动在低频区域的能量
- `freq_high_heatmap.png`: 高频残差热力图，反映被剔除高频能量的空间位置

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

## 许可证

本项目仅用于学术研究目的。