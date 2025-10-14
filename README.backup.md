(备份) 原 README.md 内容:

````markdown
# SA-MOO Adversarial Attack & RobustBench 评测工具

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
... (内容略，与原 README.md 后半部分相同)
````