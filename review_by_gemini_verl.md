# Verl (Qwen3 + GRPO) 环境避坑与安装指南

### 拜托 Gemini 老师总结的跑通 verl 训 qwen3 过程中遇到的困难
本文档详细记录了在 **A800/H800** 等高性能显卡上，配置支持 **Qwen3**、**GRPO 算法** 以及 **vLLM V1 引擎** 的 Verl 训练环境的全过程。

本文档包含两部分：
1.  **踩坑复盘**：详细记录了我们在配置过程中遇到的五大核心困难及其解决方案。
2.  **最终安装方案**：经过验证的、可直接执行的“黄金配置”脚本。

---

## Part 1: 困难总结与解决方案复盘

#### 在配置过程中，最大的坑在于 qwen3 需要 vLLM 大于等于 0.8.5。参考某位杭州大厂人才计划大满贯落叶哥和顶A大满贯压缩哥的意见，用 vllm 0.11。

我们主要遭遇了以下五大技术障碍：

### 1. “依赖地狱” (Dependency Hell)
* **现象**：`pip` 报错依赖冲突，无法解析版本。
* **原因**：
    * `verl` 核心代码强制要求 `numpy < 2.0`（为了兼容旧代码）。
    * `vllm 0.11.0` 依赖 `opencv-python-headless >= 4.11`，而新版 OpenCV 强制依赖 `numpy >= 2.0`。
    * 三者形成了版本死锁：Verl 要旧 Numpy，OpenCV 要新 Numpy，vLLM 要新 OpenCV。
* **✅ 解决方案：强制锁定法**
    * 我们选择**牺牲 OpenCV 的版本要求**来保全 `verl` 和 `vllm`。
    * 操作：`pip install "numpy<2.0.0" "opencv-python-headless<4.10"`。
    * 结果：虽然 pip 会报红色的冲突警告，但在纯文本训练（Qwen3）场景下，OpenCV 版本不匹配完全不影响 vLLM 运行。（但好像qwen3-vl是真没招了）

### 2. PyTorch 与 vLLM 的版本对齐
* **现象**：`flash-attn` 报错 `undefined symbol`（符号丢失）或 vLLM 无法启动。
* **原因**：
    * 手动安装的 PyTorch（如 2.5.1）与 `vllm 0.11.0` 预编译包所依赖的 PyTorch 版本（2.8.0）不一致。
    * `flash-attn` 如果使用预编译包，往往是针对旧版 PyTorch 的，无法在 PyTorch 2.8 上运行。
* **✅ 解决方案：跟随 vLLM 自动安装**
    * 放弃手动指定 PyTorch 版本，直接安装 `vllm==0.11.0`，让它自动拉取匹配的 `torch==2.8.0`。
    * **最后**再编译 `flash-attn`，确保它链接到当前环境的 PyTorch 2.8 动态库。

### 3. 代码与库的“时空错位”
* **现象**：`ImportError: cannot import name 'get_tcp_uri'` 或 `process_weights_after_loading`。
* **原因**：
    * 使用的 `verl 0.7.0.dev0` 是开发版代码，引用了 vLLM 极新版本（0.11+）才有的 API。
    * 最开始尝试使用的 `vllm 0.6.3` 太旧，缺少这些函数。我们尝试修改代码（try-except），但发现缺失的组件太多，无法修补。
* **✅ 解决方案：该升就升**
    * 放弃兼容旧版 vLLM，直接升级到 **`vllm 0.11.0`**，彻底解决了 API 缺失问题。
    * 同时还原 `verl` 源代码为纯净版（`git checkout .`），不再需要任何魔改补丁。


### 4. 运行时冲突：vLLM V1 引擎 vs PyTorch (最隐蔽的坑)
* **现象**：启动时报错 `AssertionError: Expandable segments are not compatible with memory pool` 或 `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False`。
* **原因**：
    * `vllm 0.11` 默认启用 V1 引擎，其**内存池设计**与 PyTorch 的 `expandable_segments:True` 功能互斥。
    * 如果禁用 V1 (`VLLM_USE_V1=0`)，`verl` 代码中硬编码的 `AsyncLLM` 类又会报错（因为它只存在于 V1）。
* **✅ 解决方案：双向奔赴**
    * **开启 V1**：`export VLLM_USE_V1=1`（满足 verl 代码要求）。
    * **禁用 PyTorch 扩展段**：`unset PYTORCH_CUDA_ALLOC_CONF`（满足 vLLM V1 引擎要求）。

---

## Part 2: 最终成功的环境安装指南

### 核心版本快照
* **Python**: 3.11
* **CUDA**: 12.6.1
* **vLLM**: 0.11.0 (支持 Qwen3 FP8 修复)
* **PyTorch**: 2.8.0 (由 vLLM 自动安装)
* **Flash Attention**: 2.8.3 (源码编译)
* **Numpy**: 1.26.4

### 1. 初始化 Conda 环境
```bash
conda create -n verl1 python=3.11 -y
conda activate verl1
```
### 2. 安装 CUDA 依赖 (官方源)
为了最大化兼容 A800 性能，安装 CUDA 12.6：
```bash
conda install cudnn -c nvidia -y
conda install cuda -c nvidia/label/cuda-12.6.1 -y
```
### 3. 安装 vLLM (核心步骤)
直接安装 vLLM，它会自动安装匹配的 PyTorch 2.8。使用清华源加速：
```bash
pip install vllm==0.11.0 -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
### 4. 锁定关键依赖 (解决冲突)
Verl 需要旧版 Numpy，而 vLLM 默认会拉取新版。我们需要手动降级 Numpy 并锁定 OpenCV 版本以避免冲突。
```bash
# 1. 安装 verl 所需的其他依赖
pip install transformers deepspeed accelerate datasets -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# 2. 强制锁定 Numpy 和 OpenCV
# 注意：忽略 pip 关于 vllm 依赖 opencv>=4.11 的红色报错，这是预期的。
pip install "numpy<2.0.0" "opencv-python-headless<4.10" -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
### 5. 安装 Verl 源码
Verl 需要旧版 Numpy，而 vLLM 默认会拉取新版。我们需要手动降级 Numpy 并锁定 OpenCV 版本以避免冲突。
```bash
cd ~/verl
pip install -e .
```
### 6. 编译 Flash Attention (最后一步)
这一步必须最后做，确保它链接到正确的 PyTorch 2.8：
```bash
export MAX_JOBS=8
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
### 7. 启动脚本配置 (关键)
在运行训练脚本（如 test_grpo_single_card.sh）时，包含以下配置以防止 vLLM V1 引擎崩溃：
```bash
#!/bin/bash

# 1. 显式开启 vLLM V1 引擎 (Verl 代码依赖此引擎)
export VLLM_USE_V1=1

# 2. 【至关重要】禁用 PyTorch Expandable Segments
# vLLM V1 的内存池与此功能冲突，必须彻底清除该变量
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=""

# 3. 指定 Attention 后端 (可选，推荐显式指定，或注释掉由 vllm 自动检测)
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ... 你的 python 启动命令 ...
python3 -m verl.trainer.main_ppo ...
```