# config.py
# 配置管理模块：支持多种配置来源
# 优先级：命令行参数 > 环境变量 > 配置文件 > 默认值

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import yaml

# 加载环境变量
load_dotenv()

class Config:
    """
    配置管理类，支持从多个来源读取配置参数

    优先级顺序（从高到低）：

    1. 命令行参数
    2. 环境变量
    3. YAML配置文件
    4. 默认值
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_file: YAML配置文件路径，如果为None则使用默认路径
        """
        self.config_file = config_file or self._find_config_file()
        self.yaml_config = self._load_yaml_config()
        self.args = self._parse_args()

        # 初始化所有配置参数
        self._init_config()

    def _find_config_file(self) -> Optional[str]:
        """查找配置文件"""
        # 优先使用环境变量指定的配置文件
        config_path = os.getenv("CONFIG_FILE")
        if config_path and Path(config_path).exists():
            return config_path

        # 查找默认配置文件
        default_configs = ["config.yaml", "config.yml", "../config.yaml", "../config.yml"]
        for config_name in default_configs:
            config_path = Path(__file__).parent.parent / config_name
            if config_path.exists():
                return str(config_path)

        return None

    def _load_yaml_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_file or not Path(self.config_file).exists():
            return {}

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_file}: {e}")
            return {}

    def _parse_args(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="SA-MOO Adversarial Attack")

        # 核心配置
        parser.add_argument("--target-image-id", type=int, help="CIFAR-10测试集图像索引")
        parser.add_argument("--model-weights-path", type=str, help="模型权重文件路径")
        parser.add_argument("--data-root-dir", type=str, help="CIFAR-10数据根目录")
        parser.add_argument("--output-dir", type=str, help="输出目录")

        # SA-MOO超参数
        parser.add_argument("--population-size", type=int, help="种群大小")
        parser.add_argument("--num-generations", type=int, help="最大代数")
        parser.add_argument("--fixed-k", type=int, help="扰动预算K")
        parser.add_argument("--crossover-prob", type=float, help="交叉概率")
        parser.add_argument("--zero-sample-prob", type=float, help="零采样概率")

        # 攻击配置
        parser.add_argument("--perturbation-mode", type=str,
                          choices=["rgb_sim", "channel", "v_channel"], help="扰动模式")
        parser.add_argument("--targeted", action="store_true", help="是否为目标攻击")
        parser.add_argument("--target-class-id", type=int, help="目标类别ID")

        # 连续扰动配置
        parser.add_argument("--enable-continuous-perturbation", action="store_true", help="启用连续扰动")
        parser.add_argument("--continuous-lower-bound", type=float, help="连续扰动下界")
        parser.add_argument("--continuous-upper-bound", type=float, help="连续扰动上界")
        parser.add_argument("--continuous-decimal-places", type=int, help="连续扰动小数位数")

        # 可视化配置
        parser.add_argument("--enable-visualization", action="store_true", help="启用可视化")
        parser.add_argument("--visualize-interval", type=int, help="可视化间隔")

        # 动态配置
        parser.add_argument("--dominance-config", type=str, help="动态支配关系配置（JSON字符串或文件路径）")
        parser.add_argument("--dynamic-params", type=str, help="动态参数调节配置（JSON字符串或文件路径）")
        parser.add_argument("--edge-guidance", type=str, help="边缘引导配置（JSON字符串或文件路径）")

        # 其他
        parser.add_argument("--config", type=str, help="指定配置文件路径")

        return parser.parse_args()

    def _get_config_value(self, key: str, default_value: Any, value_type: type = str) -> Any:
        """
        从多个来源获取配置值，按优先级返回

        Args:
            key: 配置键名
            default_value: 默认值
            value_type: 值类型

        Returns:
            配置值
        """
        # 1. 命令行参数优先级最高
        arg_key = key.replace('_', '-')
        if hasattr(self.args, arg_key) and getattr(self.args, arg_key) is not None:
            return getattr(self.args, arg_key)

        # 2. 环境变量
        env_key = f"SA_MOO_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            if value_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            return value_type(env_value)

        # 3. YAML配置文件
        if key in self.yaml_config:
            value = self.yaml_config[key]
            if value_type == bool and isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return value_type(value) if value is not None else default_value

        # 4. 默认值
        return default_value

    def _parse_external_config(self, raw_value: Optional[str], description: str) -> Dict[str, Any]:
        """解析外部配置，可以是JSON字符串或文件路径。"""
        if not raw_value:
            return {}

        candidate_path = Path(raw_value)
        if candidate_path.exists():
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    if candidate_path.suffix.lower() in {".yml", ".yaml"}:
                        return yaml.safe_load(f) or {}
                    return json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to load {description} from file '{raw_value}': {exc}") from exc

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {description} JSON string: {exc}") from exc

    @staticmethod
    def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典，不修改输入字典。"""
        if not overrides:
            return base

        result = copy.deepcopy(base)
        stack: List[tuple] = [(result, overrides)]
        while stack:
            current_base, current_override = stack.pop()
            for key, value in current_override.items():
                if isinstance(value, dict) and isinstance(current_base.get(key), dict):
                    stack.append((current_base[key], value))
                else:
                    current_base[key] = copy.deepcopy(value)
        return result

    def _load_dominance_config(self) -> Dict[str, Any]:
        """加载动态支配关系配置。"""
        default_config: Dict[str, Any] = {
            "rules": [
                {
                    "when": {"is_adversarial": True, "other_is_adversarial": False},
                    "prefer": "self"
                },
                {
                    "when": {"is_adversarial": False, "other_is_adversarial": True},
                    "prefer": "other"
                },
                {
                    "when": {"is_adversarial": True, "other_is_adversarial": True},
                    "metrics": [
                        {"name": "l2_norm", "goal": "min", "tolerance": 0.0},
                        {"name": "l0_norm", "goal": "min", "tolerance": 0.0}
                    ]
                },
                {
                    "when": {"is_adversarial": False, "other_is_adversarial": False},
                    "metrics": [
                        {"name": "loss", "goal": "min", "tolerance": 0.0},
                        {"name": "l2_norm", "goal": "min", "tolerance": 0.0}
                    ]
                }
            ],
            "tie_breakers": [
                {"name": "l0_norm", "goal": "min", "tolerance": 0.0}
            ]
        }

        dominance_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("dominance") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            dominance_config = self._deep_update(dominance_config, yaml_override)

        env_override = os.getenv("SA_MOO_DOMINANCE_CONFIG")
        if env_override:
            dominance_config = self._deep_update(dominance_config, self._parse_external_config(env_override, "dominance config"))

        arg_override = getattr(self.args, "dominance_config", None)
        if arg_override:
            dominance_config = self._deep_update(dominance_config, self._parse_external_config(arg_override, "dominance config"))

        return dominance_config

    def _load_dynamic_parameters(self) -> Dict[str, Any]:
        """加载动态参数调节配置。"""
        default_config: Dict[str, Any] = {
            "mutation_probability": {
                "schedule": "linear",
                "targeted": {"start": 0.2, "end": 0.01},
                "non_targeted": {"start": 0.4, "end": 0.01},
                "min": 0.0,
                "max": 1.0
            },
            "crossover_prob": {
                "schedule": "constant",
                "value": self.CROSSOVER_PROB
            },
            "zero_sample_prob": {
                "schedule": "constant",
                "value": self.ZERO_SAMPLE_PROB
            }
        }

        dynamic_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("dynamic_parameters") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            dynamic_config = self._deep_update(dynamic_config, yaml_override)

        env_override = os.getenv("SA_MOO_DYNAMIC_PARAMS")
        if env_override:
            dynamic_config = self._deep_update(dynamic_config, self._parse_external_config(env_override, "dynamic parameter config"))

        arg_override = getattr(self.args, "dynamic_params", None)
        if arg_override:
            dynamic_config = self._deep_update(dynamic_config, self._parse_external_config(arg_override, "dynamic parameter config"))

        return dynamic_config

    def _load_edge_guidance_config(self) -> Dict[str, Any]:
        """加载边缘引导配置。"""
        default_config: Dict[str, Any] = {
            "enabled": False,
            "method": "sobel",
            "gaussian_sigma": 0.8,
            "canny_sigma": 1.0,
            "exponent": 1.5,
            "uniform_mix": 0.15,
            "min_value": 1e-4,
            "multi_scale": {
                "enabled": False,
                "scales": [1.0, 0.5, 0.25],
                "weights": None,
                "gaussian_sigmas": None,
                "combine": "mean",
                "anti_aliasing": True,
            },
            "semantic": {
                "enabled": False,
                "path": None,
                "array": None,
                "weight": 0.5,
                "blend_mode": "multiply",
                "normalize": True,
                "invert": False,
                "blur_sigma": 0.0,
                "exponent": 1.0,
                "clip_low": None,
                "clip_high": None,
                "fail_on_missing": False,
            },
            "professional_preprocessing": {
                "enabled": False,
                "edge_map_path": None,
                "edge_normalize": False,
                "edge_clip_low": None,
                "edge_clip_high": None,
                "semantic_map_path": None,
                "semantic_normalize": True,
                "semantic_clip_low": None,
                "semantic_clip_high": None,
                "fail_on_missing": False,
                "semantic_model": {
                    "enabled": False,
                    "name": "deeplabv3_resnet50",
                    "device": "auto",
                    "output": "max_prob",
                    "target_class": None,
                    "smooth_sigma": 0.0,
                    "entropy_eps": 1e-6,
                },
                "edge_model": {
                    "enabled": False,
                    "type": "semantic_gradient",
                    "device": "auto",
                    "normalize": True,
                    "smooth_sigma": 0.0,
                    "canny_sigma": 1.0,
                    "semantic_output": "max_prob",
                    "semantic_smooth_sigma": 0.0,
                    "entropy_eps": 1e-6,
                    "target_class": None,
                },
            },
        }

        edge_config = copy.deepcopy(default_config)

        yaml_override = self.yaml_config.get("edge_guidance") if isinstance(self.yaml_config, dict) else None
        if isinstance(yaml_override, dict):
            edge_config = self._deep_update(edge_config, yaml_override)

        env_override = os.getenv("SA_MOO_EDGE_GUIDANCE")
        if env_override:
            edge_config = self._deep_update(edge_config, self._parse_external_config(env_override, "edge guidance config"))

        arg_override = getattr(self.args, "edge_guidance", None)
        if arg_override:
            edge_config = self._deep_update(edge_config, self._parse_external_config(arg_override, "edge guidance config"))

        return edge_config

    def _init_config(self):
        """初始化所有配置参数"""
        # ==================== 核心配置 ====================
        self.TARGET_IMAGE_ID: int = self._get_config_value("target_image_id", 7780, int)

        # 模型权重路径处理 - 优先从环境变量读取
        weights_path = os.getenv("MODEL_WEIGHTS_PATH") or self._get_config_value("model_weights_path", "cifar10_resnet18.pth", str)
        self.MODEL_WEIGHTS_PATH: Path = Path(weights_path)
        if not self.MODEL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {self.MODEL_WEIGHTS_PATH.absolute()}.")

        # 数据和输出目录 - 优先从环境变量读取
        data_root = os.getenv("DATA_ROOT_DIR") or self._get_config_value("data_root_dir", "./data", str)
        self.DATA_ROOT_DIR: Path = Path(data_root)

        output_dir = os.getenv("OUTPUT_DIR") or self._get_config_value("output_dir", "../output", str)
        self.OUTPUT_DIR: Path = Path(output_dir)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Torch环境变量
        torch_home = os.getenv("TORCH_HOME")
        lpips_cache = os.getenv("LPIPS_CACHE_DIR")
        if torch_home:
            self.TORCH_HOME: Path = Path(torch_home).resolve()
        else:
            self.TORCH_HOME = Path.home() / ".cache" / "torch"
        if lpips_cache:
            self.LPIPS_CACHE_DIR: Path = Path(lpips_cache).resolve()
        else:
            self.LPIPS_CACHE_DIR = Path.home() / ".cache" / "torch" / "lpips"

        os.environ["TORCH_HOME"] = str(self.TORCH_HOME)
        os.environ["LPIPS_CACHE_DIR"] = str(self.LPIPS_CACHE_DIR)

        # ==================== SA-MOO 超参数 ====================
        self.POPULATION_SIZE: int = self._get_config_value("population_size", 2, int)
        self.NUM_GENERATIONS: int = self._get_config_value("num_generations", 1000, int)
        self.FIXED_K: int = self._get_config_value("fixed_k", 24, int)
        self.CROSSOVER_PROB: float = self._get_config_value("crossover_prob", 0.1, float)
        self.ZERO_SAMPLE_PROB: float = self._get_config_value("zero_sample_prob", 0.3, float)

        # ==================== 攻击配置 ====================
        self.PERTURBATION_MODE: str = self._get_config_value("perturbation_mode", "rgb_sim", str)
        self.IS_TARGETED_ATTACK: bool = self._get_config_value("is_targeted_attack", False, bool)
        self.TARGET_CLASS_ID: Optional[int] = self._get_config_value("target_class_id", 5, int)

        # ==================== 连续扰动配置 ====================
        self.ENABLE_CONTINUOUS_PERTURBATION: bool = self._get_config_value("enable_continuous_perturbation", False, bool)
        self.CONTINUOUS_LOWER_BOUND: float = self._get_config_value("continuous_lower_bound", -1.0, float)
        self.CONTINUOUS_UPPER_BOUND: float = self._get_config_value("continuous_upper_bound", 1.0, float)
        self.CONTINUOUS_DECIMAL_PLACES: int = self._get_config_value("continuous_decimal_places", 2, int)

        # ==================== 可视化配置 ====================
        self.ENABLE_VISUALIZATION: bool = self._get_config_value("enable_visualization", False, bool)
        self.VISUALIZE_INTERVAL: int = self._get_config_value("visualize_interval", 500, int)

        # ==================== 真实世界鲁棒性配置 ====================
        self.ENABLE_REAL_WORLD_ROBUSTNESS: bool = self._get_config_value("enable_real_world_robustness", False, bool)
        self.JPEG_QUALITY: int = self._get_config_value("jpeg_quality", 75, int)
        self.ENABLE_RESIZE_PREPROCESSING: bool = self._get_config_value("enable_resize_preprocessing", True, bool)
        self.RESIZE_SCALE: float = self._get_config_value("resize_scale", 2.0, float)

        # ==================== 动态配置 ====================
        self.DOMINANCE_CONFIG: Dict[str, Any] = self._load_dominance_config()
        self.DYNAMIC_PARAMETER_CONFIG: Dict[str, Any] = self._load_dynamic_parameters()
        self.EDGE_GUIDANCE_CONFIG: Dict[str, Any] = self._load_edge_guidance_config()

    def print_config(self):
        """打印当前配置"""
        print("=== Current Configuration ===")
        print(f"Target Image ID: {self.TARGET_IMAGE_ID}")
        print(f"Model Weights: {self.MODEL_WEIGHTS_PATH}")
        print(f"Data Root: {self.DATA_ROOT_DIR}")
        print(f"Output Dir: {self.OUTPUT_DIR}")
        print(f"Perturbation Mode: {self.PERTURBATION_MODE}")
        print(f"Targeted Attack: {self.IS_TARGETED_ATTACK}")
        if self.IS_TARGETED_ATTACK:
            print(f"Target Class ID: {self.TARGET_CLASS_ID}")
        print(f"Continuous Perturbation: {self.ENABLE_CONTINUOUS_PERTURBATION}")
        if self.ENABLE_CONTINUOUS_PERTURBATION:
            print(f"Continuous Bounds: [{self.CONTINUOUS_LOWER_BOUND}, {self.CONTINUOUS_UPPER_BOUND}]")
            print(f"Decimal Places: {self.CONTINUOUS_DECIMAL_PLACES}")
        print(f"Population Size: {self.POPULATION_SIZE}")
        print(f"Generations: {self.NUM_GENERATIONS}")
        print(f"Fixed K: {self.FIXED_K}")
        print(f"Visualization: {self.ENABLE_VISUALIZATION}")
        print(f"Real World Robustness: {self.ENABLE_REAL_WORLD_ROBUSTNESS}")
        print("=" * 30)

# 创建全局配置实例
config = Config()

# 为了向后兼容，将配置值导出为模块级变量
TARGET_IMAGE_ID = config.TARGET_IMAGE_ID
MODEL_WEIGHTS_PATH = config.MODEL_WEIGHTS_PATH
DATA_ROOT_DIR = config.DATA_ROOT_DIR
OUTPUT_DIR = config.OUTPUT_DIR
TORCH_HOME = config.TORCH_HOME
LPIPS_CACHE_DIR = config.LPIPS_CACHE_DIR
POPULATION_SIZE = config.POPULATION_SIZE
NUM_GENERATIONS = config.NUM_GENERATIONS
FIXED_K = config.FIXED_K
CROSSOVER_PROB = config.CROSSOVER_PROB
ZERO_SAMPLE_PROB = config.ZERO_SAMPLE_PROB
PERTURBATION_MODE = config.PERTURBATION_MODE
IS_TARGETED_ATTACK = config.IS_TARGETED_ATTACK
TARGET_CLASS_ID = config.TARGET_CLASS_ID
ENABLE_CONTINUOUS_PERTURBATION = config.ENABLE_CONTINUOUS_PERTURBATION
CONTINUOUS_LOWER_BOUND = config.CONTINUOUS_LOWER_BOUND
CONTINUOUS_UPPER_BOUND = config.CONTINUOUS_UPPER_BOUND
CONTINUOUS_DECIMAL_PLACES = config.CONTINUOUS_DECIMAL_PLACES
ENABLE_VISUALIZATION = config.ENABLE_VISUALIZATION
VISUALIZE_INTERVAL = config.VISUALIZE_INTERVAL
ENABLE_REAL_WORLD_ROBUSTNESS = config.ENABLE_REAL_WORLD_ROBUSTNESS
JPEG_QUALITY = config.JPEG_QUALITY
ENABLE_RESIZE_PREPROCESSING = config.ENABLE_RESIZE_PREPROCESSING
RESIZE_SCALE = config.RESIZE_SCALE
DOMINANCE_CONFIG = config.DOMINANCE_CONFIG
DYNAMIC_PARAMETER_CONFIG = config.DYNAMIC_PARAMETER_CONFIG
EDGE_GUIDANCE_CONFIG = config.EDGE_GUIDANCE_CONFIG