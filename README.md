# SA-MOO Adversarial Attack

This is an improved reproduction of the paper "Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation" (SA-MOO) in Python.

## Features

- **Three Perturbation Modes**:
  - `rgb_sim`: Modify all RGB channels of selected pixels simultaneously
  - `channel`: Modify a single RGB channel (original paper method)
  - `v_channel`: Perturb only the V (brightness) channel in HSV color space

- **Two Perturbation Types**:
  - **Discrete Perturbation**: Perturbation values chosen from {-1, 0, 1} (default)
  - **Continuous Perturbation**: Perturbation values randomly selected from a specified range with n decimal places

- **Attack Types**:
  - Untargeted Attack: Cause the model to misclassify, without specifying a particular class
  - Targeted Attack: Cause the model to classify the image as a specified target class

- **Multi-Objective Optimization**:
  - Minimize L2 norm (perturbation magnitude)
  - Minimize L0 norm (number of perturbed pixels/channels)
  - Maximize adversarial success rate

- **Semantic Edge Guidance**: Generate edge weights using Sobel/Scharr/Prewitt/Canny operators, with optional multi-scale fusion and semantic heatmap weighting, making initialization and mutation closer to key structures.

## Project Structure

```
src/
├── config/
│   └── config.py          # Global configuration parameters
├── utils/
│   ├── edge_guidance.py   # Semantic edge weight calculation
│   └── logger.py          # Logging utilities
├── data/
│   └── data_loader.py     # Data loading and model initialization
├── core/
│   ├── objectives.py      # Objective functions and dominance relations
│   ├── evolutionary_operators.py  # Evolutionary operators (initialization, crossover, mutation, selection)
│   └── dynamic_parameters.py      # Dynamic hyperparameter scheduler
├── visualization/
│   └── visualization.py   # Visualization and result saving
└── main.py                # Main execution file
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Modify the following key parameters in `src/config/config.py`:

- `TARGET_IMAGE_ID`: Index of the target image in the CIFAR-10 test set
- `MODEL_WEIGHTS_PATH`: Path to pre-trained ResNet18 model weights
- `PERTURBATION_MODE`: Perturbation mode selection
- `IS_TARGETED_ATTACK`: Whether it is a targeted attack
- `FIXED_K`: Perturbation budget (L0 constraint)
- `NUM_GENERATIONS`: Number of evolutionary generations

### Continuous Perturbation Configuration

- `ENABLE_CONTINUOUS_PERTURBATION`: Whether to enable continuous perturbation (default False)
- `CONTINUOUS_LOWER_BOUND`: Lower bound for continuous perturbation (default -1.0)
- `CONTINUOUS_UPPER_BOUND`: Upper bound for continuous perturbation (default 1.0)
- `CONTINUOUS_DECIMAL_PLACES`: Number of decimal places (default 2)

When continuous perturbation is enabled, perturbation values will be randomly selected from the [lower bound, upper bound] range and rounded to the specified decimal places.

### Dynamic Dominance Configuration

The `dominance` section in `config.yaml` can be used to define adaptive dominance relation rules. Each rule includes:

- `when`: Trigger condition, supports boolean combinations like `is_adversarial` and `other_is_adversarial`.
- `prefer`: Directly specify the preferred individual (`self` / `other`).
- `metrics`: If `prefer` is omitted, compare specified metrics in order, supporting `goal: min|max` and `tolerance` tolerance.
- `tie_breakers`: Global fallback comparison metrics for additional sorting when all rules are undecided.

The default configuration reproduces the dominance relation in the paper "success before failure, L2 priority". You can build more complex multi-objective decision logic by adding/removing rules or changing priorities without modifying the code.

Additionally, you can override using command line or environment variables:

```bash
python run.py --dominance-config dominance_rules.json
# or
export SA_MOO_DOMINANCE_CONFIG='{"rules": [...]}'
```

### Dynamic Parameter Adjustment

The `dynamic_parameters` section allows key hyperparameters to automatically adjust with evolutionary generations. Supported schedulers:

- `constant`: Constant value.
- `linear`: Linear interpolation, supporting separate `start`/`end` for targeted and untargeted attacks.
- `exponential`: Exponential transition (effective when both `start` and `end` are greater than 0).
- `piecewise`: Piecewise constant scheduling, example:

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

Common adjustable parameters:

- `mutation_probability`
- `crossover_prob`
- `zero_sample_prob`
- `fixed_k` (experimental support)

Similarly, inject new schedules via CLI / environment variables:

```bash
python run.py --dynamic-params dynamic_schedule.yaml
export SA_MOO_DYNAMIC_PARAMS='{"mutation_probability": {"schedule": "linear", "start": 0.3, "end": 0.05}}'
```

### Semantic Edge Guidance

The `edge_guidance` section makes initialization and mutation more biased towards image semantic contours:

- `enabled`: Whether to enable edge guidance. Falls back to uniform sampling when disabled.
- `method`: Edge detection operator, supports `sobel` / `scharr` / `prewitt` / `canny`.
- `gaussian_sigma`: Gaussian smoothing strength before edge detection.
- `exponent`: Exponential amplification coefficient for edge strength, increasing can strengthen preference for strong edges.
- `uniform_mix`: Mixing ratio with uniform distribution, avoiding over-concentration; range [0,1].
- `min_value`: Lower limit added before normalization to prevent zero probability.
- `canny_sigma`: Effective only when `method=canny`, smoothing coefficient for edge detection.
- `multi_scale`: Multi-scale configuration set:
  - `enabled`: Whether to enable multi-scale fusion.
  - `scales`: List of scales (scaling factors) participating in fusion, e.g., `[1.0, 0.75, 0.5]`.
  - `sigma`: Additional Gaussian smoothing for each scale; can be a single value or a list equal in length to `scales`.
  - `aggregation`: Fusion strategy, supports `max` / `mean` / `sum`.
- `semantic`: Semantic map fusion configuration:
  - `enabled`: Whether to load semantic segmentation maps and participate in fusion.
  - `map_path`: File path to semantic segmentation probability map or mask.
  - `weight`: Fusion strength, larger values make semantic guidance more prominent.
  - `normalize`: Whether to normalize the semantic map to [0,1].
  - `resize_mode`: Interpolation method when aligning semantic map to input resolution, supports `bilinear` / `nearest`.
- `professional_preprocessing`: Use professional model outputs for preprocessing:
  - `enabled`: Whether to enable professional model output to override internal calculations.
  - `edge_map_path`: File path to professional edge detection output (supports png/jpg/npy/npz).
  - `edge_normalize`: Whether to perform 0-1 normalization on external edge maps.
  - `semantic_map_path`: File path to professional semantic segmentation output.
  - `semantic_normalize`: Whether to perform 0-1 normalization on external semantic maps (still processable per `semantic` config afterward).
  - `fail_on_missing`: Whether to throw an exception when external files are not found (default skip).
  - `semantic_model`: Directly call professional semantic segmentation models at runtime (currently built-in `torchvision` weights, e.g., `deeplabv3_resnet50`).
    - `name`: Model name, supports `deeplabv3_resnet50` / `fcn_resnet50` / `lraspp_mobilenet_v3_large`.
    - `device`: `auto` / `cpu` / `cuda`, auto-select by default.
    - `output`: Output type (`max_prob`, `entropy`, `target_class`, `top2_gap`).
  - `edge_model`: Use professional models or semantic maps to generate high-quality edge maps.
    - `type`: `semantic_gradient` (default), `semantic_entropy`, `torchvision_deeplabv3_resnet50`, `canny_rgb`.
    - `normalize` / `smooth_sigma`: Control normalization and smoothing after generation.

Configuration example:

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

When semantic guidance is enabled, the program will automatically generate a single `semantic_guidance_overview.png` debug image in the output directory (including grayscale reference, basic edges, semantic map, fusion results, and final sampling weights) for quick understanding of the guidance process.

Similarly, override using command line or environment variables:

```bash
python run.py --edge-guidance edge_config.yaml
export SA_MOO_EDGE_GUIDANCE='{"enabled": true, "method": "canny", "canny_sigma": 1.2}'
```

## Environment Variables

Set via `.env` file or system environment variables:

- `MODEL_WEIGHTS_PATH`: Model weights file path
- `DATA_ROOT_DIR`: CIFAR-10 dataset path
- `OUTPUT_DIR`: Output results save directory
- `TORCH_HOME`: PyTorch cache directory
- `LPIPS_CACHE_DIR`: LPIPS library cache directory
- `SA_MOO_DOMINANCE_CONFIG`: Points to JSON/YAML file or direct JSON string for overriding dominance relations
- `SA_MOO_DYNAMIC_PARAMS`: Points to JSON/YAML file or direct JSON string for overriding dynamic parameter scheduling
- `SA_MOO_EDGE_GUIDANCE`: Points to JSON/YAML file or direct JSON string for configuring edge guidance parameters

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration
python run.py
```

### Run Experiments with Configuration Files

#### Method 1: Use Preset Configuration Files
```bash
# RGB simultaneous perturbation experiment
python run_experiment.py rgb_sim_config.yaml

# V channel (brightness) perturbation experiment
python run_experiment.py v_channel_config.yaml

# Targeted attack experiment
python run_experiment.py targeted_attack_config.yaml

# Quick test
python run_experiment.py quick_test_config.yaml
```

#### Method 2: Use Command Line Arguments
```bash
# Specify parameters directly (overrides default config)
python run.py --perturbation-mode v_channel --fixed-k 6 --num-generations 500

# Targeted attack
python run.py --targeted --target-class-id 3 --fixed-k 12

# Enable continuous perturbation
python run.py --enable-continuous-perturbation --continuous-lower-bound -0.5 --continuous-upper-bound 0.5 --continuous-decimal-places 3

# Specify custom dominance relations and dynamic parameter scheduling
python run.py --dominance-config dominance_rules.yaml --dynamic-params dynamic_schedule.json
```

#### Method 3: Use Environment Variables
```bash
# Set environment variables
export SA_MOO_PERTURBATION_MODE=v_channel
export SA_MOO_FIXED_K=6
python run.py
```

### Configuration File Management

The project includes the following configuration files:

- **`complete_config.yaml`** - **Complete Configuration Reference** (includes all parameters and detailed explanations)
- **`config.yaml`** - Default configuration file
- **`experiment_configs/`** - Experiment-specific configuration folder

#### Using Complete Configuration

```bash
# View all available parameters (complete config)
python run_experiment.py complete_config.yaml

# Create new experiment based on complete config template
cp complete_config.yaml experiment_configs/my_experiment.yaml
# Edit parameters in my_experiment.yaml
python run_experiment.py my_experiment.yaml
```

You can create multiple configuration files in the `experiment_configs/` directory:

```yaml
# Custom configuration example
target_image_id: 1000
perturbation_mode: "channel"
fixed_k: 18
num_generations: 2000
is_targeted_attack: true
target_class_id: 7
```

Then use:
```bash
python run_experiment.py your_custom_config.yaml
```

### 🔧 Path Configuration

**Important**: All path configurations must now be set in the `.env` file:

```bash
# .env file example
MODEL_WEIGHTS_PATH=cifar10_resnet18.pth
DATA_ROOT_DIR=./data
OUTPUT_DIR=../attack_results
TORCH_HOME=/path/to/torch/cache
LPIPS_CACHE_DIR=/path/to/lpips/cache
```

Path configuration priority:
1. **Environment Variables** (set in `.env` file)
2. **Command Line Arguments** (e.g., `--model-weights-path`)
3. **Default Values** (if neither environment variables nor command line arguments are set)

## Output Results

The program will create a timestamp-named results directory under `OUTPUT_DIR`, containing:

- `report.txt`: Detailed execution log
- `adversarial_result.png`: Adversarial attack result comparison image
- `noise_*.png`: Noise heatmaps
- `deltaL_heatmap.png`: CIELAB color space ΔL* difference image
- `convergence.png`: L2/L0 norm convergence curves
- `final_perturbed.png`: 32x32 perturbed image (can be directly used for testing)
- `semantic_guidance_overview.png`: If semantic guidance is enabled, additionally saves a comprehensive diagnostic image including grayscale, basic edges, semantic map, and fusion weights

## Evaluation Metrics

- **L2 Norm**: Euclidean norm of the perturbation
- **L0 Norm**: Number of non-zero perturbed pixels/channels
- **PSNR**: Peak Signal-to-Noise Ratio
- **LPIPS**: Perceptual Loss
- **ΔE00**: Color difference in CIELAB color space

## Notes

1. Ensure the model weights file exists and the path is correct
2. For `v_channel` mode, it is recommended to set `FIXED_K=6`
3. Targeted attacks require setting `TARGET_CLASS_ID`
4. Visualization features require graphical interface support

## Citation

If using this code, please cite the original paper:

```
Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation
```

## MIT License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

