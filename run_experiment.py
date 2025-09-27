# run_experiment.py
# 实验运行脚本
# 支持使用不同的配置文件运行实验
# 可以在下方选取./experiment_configs路径下的yaml

import sys
import os
from pathlib import Path

def run_experiment(config_file: str):
    """
    使用指定配置文件运行实验

    Args:
        config_file: 配置文件路径
    """
    # 设置环境变量指定配置文件
    os.environ["CONFIG_FILE"] = config_file

    # 将src目录添加到Python路径
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

    # 导入并运行主函数
    try:
        from main import main
        main()
    except ImportError as e:
        print(f"Error importing main module: {e}")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)

def main():
    """主函数：解析命令行参数并运行实验"""
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_file>")
        print("\nAvailable config files in experiment_configs/:")
        config_dir = Path(__file__).parent / "experiment_configs"
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file.name}")
        print("\nOr specify full path to any config file")
        sys.exit(1)

    config_file = sys.argv[1]

    # 如果是相对路径，尝试在experiment_configs目录中查找
    if not Path(config_file).is_absolute() and not config_file.startswith("./") and not config_file.startswith("../"):
        possible_path = Path(__file__).parent / "experiment_configs" / config_file
        if possible_path.exists():
            config_file = str(possible_path)
        elif not Path(config_file).exists():
            print(f"Error: Config file '{config_file}' not found")
            print(f"Searched in: {possible_path}")
            sys.exit(1)

    print(f"Running experiment with config: {config_file}")
    run_experiment(config_file)

if __name__ == "__main__":
    main()