# run.py
# 项目启动脚本
# 从项目根目录运行此脚本来执行SA-MOO攻击
# 默认使用同目录下的config.yaml参数

import sys
from pathlib import Path

# 将src目录添加到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# 导入并运行主函数
from main import main

if __name__ == "__main__":
    main()