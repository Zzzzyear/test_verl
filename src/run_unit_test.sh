#!/bin/bash
# 路径：src/run_tests.sh

# ===================================================
# EGPO 算法逻辑验证脚本 (Fixed for src/tests structure)
# ===================================================

# 1. 锁定脚本所在目录 (即 .../workspace/EGPO/src)
SRC_DIR=$(cd "$(dirname "$0")"; pwd)
echo "📂 Working Directory: $SRC_DIR"

# 2. 进入 src 目录
cd "$SRC_DIR"

# 3. 设置 PYTHONPATH
# 将当前目录 (src) 加入 Python 路径
# 这样代码里可以直接 "import egpo" 和 "import tests"
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
echo "🔗 PYTHONPATH set to include src."

# 4. 运行测试
echo "========================================"
echo "🧪 Running EGPO Core Logic Tests..."
echo "========================================"

# [关键修正]: 
# 1. 使用模块点分法: tests.test_egpo_core (不要用斜杠)
# 2. 确保 src/tests/__init__.py 存在 (虽然 Py3 不强制，但加上更稳)
touch tests/__init__.py 

python3 -m unittest tests.test_egpo_core -v

# 5. 结果反馈
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "✅ SUCCESS! 算法逻辑验证通过。"
    echo "========================================"
else
    echo "========================================"
    echo "❌ FAILED. 请检查代码逻辑。"
    echo "========================================"
    exit 1
fi