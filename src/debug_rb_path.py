import os
from pathlib import Path
import importlib
from dotenv import load_dotenv

# 主动加载 .env 保证环境变量可用
load_dotenv(override=False)

# 读取环境变量
raw_rb_dir = os.environ.get("ROBUST_BENCH_DIR")
torch_home = os.environ.get("TORCH_HOME")
if not raw_rb_dir:
    # 回退：使用 TORCH_HOME/robustbench
    if torch_home:
        raw_rb_dir = str(Path(torch_home) / "robustbench")
robustbench_root = Path(raw_rb_dir) if raw_rb_dir else Path.cwd() / "model_cache" / "robustbench"

# 嵌套 autodetect（与 robustbench_eval 对齐）
nested_candidate = robustbench_root / "robustbench" / "models"
if not (robustbench_root / "models").exists() and nested_candidate.exists():
    robustbench_root = robustbench_root / "robustbench"

os.environ["ROBUST_BENCH_DIR"] = str(robustbench_root)
print(f"[DebugScript] Using ROBUST_BENCH_DIR: {robustbench_root}")

try:
    utils_module = importlib.import_module("robustbench.utils")
    model_zoo_module = importlib.import_module("robustbench.model_zoo")
except ModuleNotFoundError as e:
    print("robustbench 未安装: ", e)
    raise SystemExit(1)

# 尝试读取内部常量或函数
get_and_save_model = getattr(utils_module, "download_model", None) or getattr(utils_module, "get_and_save_model", None)

# 探索 model_zoo 中的模型信息结构
MODEL_LIST = getattr(model_zoo_module, "MODEL_LIST", None)
if MODEL_LIST is None:
    print("未找到 MODEL_LIST, 可能 robustbench 版本不同")
else:
    # 找到 Standard 模型条目
    std_entries = [m for m in MODEL_LIST if m.get('name') == 'Standard' and m.get('dataset') == 'cifar10' and m.get('threat_model') == 'Linf']
    if not std_entries:
        print("MODEL_LIST 中没有找到 Standard cifar10 Linf")
    else:
        entry = std_entries[0]
        gdrive_id = entry.get('gdrive_id')
        print("Standard 模型 gdrive_id:", gdrive_id)

# RobustBench 构造路径的方式： ROBUST_BENCH_DIR / 'models' / dataset / threat_model / f"{model_name}.pt"
expected = robustbench_root / 'models' / 'cifar10' / 'Linf' / 'Standard.pt'
print("[DebugScript] Expected model file:", expected)
print("[DebugScript] Exists?", expected.exists())

# 列出 utils_module 中所有可能与下载相关的可调用对象
possible_download = []
for name in dir(utils_module):
    if 'download' in name.lower() or 'get_and_save' in name.lower():
        obj = getattr(utils_module, name)
        if callable(obj):
            possible_download.append(name)
print("[DebugScript] utils 中可疑下载函数:", possible_download)

# 尝试真正调用 load_model ，但捕获它的下载行为
try:
    from robustbench.utils import load_model
    import traceback
    print("调用 load_model 前 (将尝试使用上面 ROBUST_BENCH_DIR)...")
    model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    print("load_model 成功，无需下载 (或已缓存)。")
except Exception as e:
    print("load_model 触发异常，可能是下载阶段:")
    traceback.print_exc()
