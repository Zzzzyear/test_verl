import os

def get_project_root():
    """
    自动判断当前运行环境，返回项目根目录。
    优先级: 训练服务器 -> 测试服务器 -> 当前目录递归查找
    """
    candidates = [
        "/data-store/zhaoqiannian/workspace/EGPO",
        "/data/zhaoqn/workspace/EGPO",
        os.getcwd()
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            return path
            
    # 向上递归寻找 src 目录作为锚点
    current = os.getcwd()
    while current != "/":
        if os.path.exists(os.path.join(current, "src", "egpo")):
            return current
        current = os.path.dirname(current)
    
    raise FileNotFoundError("Could not find EGPO project root. Please set PYTHONPATH.")

# 全局常量
PROJECT_ROOT = get_project_root()
MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "models")