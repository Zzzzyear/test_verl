# /data/zhaoqn/workspace/EGPO/src/scripts/prep_openr1_math_pool30k.py
# ------------------------------------------------------------
# Build OpenR1-Math Pool-30k for EGPO (RL + SFT compatible)
#
# Modes:
#   1) pure_random:
#        - control SUBSET_RATIO (default/extended)
#        - subset-internal pure random (no source cap, no bucket control)
#   2) random_source_softcap:
#        - control SUBSET_RATIO
#        - subset-internal random + source softcap (dominant cap + longtail floors)
#        - no bucket control
#   3) bucket_source_softcap:
#        - control SUBSET_RATIO
#        - enforce BUCKET_RATIO within each subset
#        - apply source softcap inside each bucket
#
# Output:
#   - math_openr1_pool30k_<mode>_train_final.parquet
#   - math_openr1_pool30k_<mode>_val_fixed.parquet
#   - recipe + uuid list
# Optionally also write canonical copies (no mode suffix).
# ------------------------------------------------------------

import os
import re
import glob
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from datasketch import MinHash, MinHashLSH


# =========================
# 0) 配置区
# =========================


'''
SEED = 42
# 三种模式：
#   "pure_random"
#   "random_source_softcap"
#   "bucket_source_softcap"
SAMPLING_MODE = "random_source_softcap"

POOL_TOTAL = 30000
VAL_SIZE = 128

# default / extended 配比（写死可复现）
SUBSET_RATIO = {"default": 0.7, "extended": 0.3}

# bucket 配比：OpenR1-Math-220k 基本没有 all_fail，所以默认不写它
# 可以改成 {"mixed": 0.5, "all_correct": 0.5} 或其它
BUCKET_RATIO = {"mixed": 0.3, "all_correct": 0.7}

# 过滤：至少要有 2 条 generation（OpenR1 常见 N=2/4）
MIN_GENERATIONS = 2

# 长度过滤（强烈建议写死）
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 4096  # 更稳吞吐可改 3072

#  reward_manager → GT 必须单值（否则 reward 噪声大/判错）
ENFORCE_SINGLE_VALUE_ANSWER = True

# 稳定性强开关：
# - True：要求 solution 或某条 generation 的 “最后一个 \\boxed{}” 能匹配 answer（轻量归一化后）
# - 会显著减少“答案抽取方式不一致”带来的 reward 噪声
CHECK_LAST_BOXED_MATCH = True
REQUIRE_ANY_BOXED_MATCH = True  # 若匹配失败则丢弃（建议 True）
REQUIRE_NONEMPTY_SOLUTION = True  # solution 为空且无法 fallback 时丢弃

# 去污染/去重
ENABLE_DECONTAM = True
ENABLE_INTERNAL_DEDUP = True

# LSH 参数
NUM_PERM = 128
LSH_THRESHOLD = 0.8

# 输出命名
OUTPUT_WITH_MODE_SUFFIX = True
ALSO_WRITE_CANONICAL_COPY = False
CANONICAL_PREFIX = "math_openr1_pool30k"

# OpenR1 原始分片目录（自动拼到 datasets/raw 下）
OPENR1_SUBDIR = "OpenR1-Math/all"

# --------- source softcap 配置（仅在 random_source_softcap / bucket_source_softcap 生效）---------
# 目标：防止 dominant source 吞掉训练分布，同时让长尾保留一定存在感（更稳泛化）
SOURCE_SOFTCAP = {
    "default": {
        "dominant": "olympiads",
        "cap_frac": 0.70,  # dominant 最多占 subset（或 bucket）比例
        "floor_abs": {     # 长尾最少保底条数（绝对值）
            "cn_contest": 200,
            "aops_forum": 200,
            "amc_aime": 100,
            "inequalities": 50,
            "number_theory": 50,
            "olympiads_ref": 20,
        },
    },
    "extended": {
        "dominant": "cn_k12",
        "cap_frac": 0.70,
        "floor_abs": {
            "olympiads": 200,
            "cn_contest": 100,
            "aops_forum": 100,
            "amc_aime": 50,
            "inequalities": 30,
            "number_theory": 30,
        },
    },
}
'''

'''
SEED = 42
SAMPLING_MODE = "bucket_source_softcap"

# 总量 10k
POOL_TOTAL = 10000
VAL_SIZE = 128

# 维持 Default 85% / Extended 15%
# 理由：1.7B 仍需 Extended 里的 K12 简单题来“润滑”训练过程
SUBSET_RATIO = {"default": 0.85, "extended": 0.15}

# 维持 Mixed 优先策略
BUCKET_RATIO = {"mixed": 0.6, "all_correct": 0.4}

MIN_GENERATIONS = 2
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 4096

ENFORCE_SINGLE_VALUE_ANSWER = True
CHECK_LAST_BOXED_MATCH = True
REQUIRE_ANY_BOXED_MATCH = True
REQUIRE_NONEMPTY_SOLUTION = True

ENABLE_DECONTAM = True
ENABLE_INTERNAL_DEDUP = True

NUM_PERM = 128
LSH_THRESHOLD = 0.8

OUTPUT_WITH_MODE_SUFFIX = True
ALSO_WRITE_CANONICAL_COPY = False
CANONICAL_PREFIX = "math_openr1_pool10k_stable"  # 改名：Stable版

OPENR1_SUBDIR = "OpenR1-Math/all"

# 【核心修改：现实可达版 Softcap】
# 基于 default sample N=2000 的统计推算，不再设虚高的 floor
SOURCE_SOFTCAP = {
    "default": {
        "dominant": "olympiads",
        # 压到 60%，给 cn_contest 和 aops 留足空间
        "cap_frac": 0.60,
        
        "floor_abs": {
            # 外推期望约 272，设 300 属于"尽量全拿但接受缺口"
            # 作用：最大化 AIME 利用率，同时不强求虚无的数量
            "amc_aime": 300,
            
            # 外推期望约 1122，设 1000
            # 作用：这是中高难度的中流砥柱，必须稳住
            "cn_contest": 1000,
            
            # 外推期望约 710，设 500
            # 作用：AoPS 题目非结构化强，留 500 条增加泛化性
            "aops_forum": 500,
            
            # 外推期望约 85，设 80
            "olympiads_ref": 80,
            
            # 小众源保底，增加分布的长尾多样性
            "inequalities": 80,
            "number_theory": 50,
        },
    },
    "extended": {
        "dominant": "cn_k12",
        "cap_frac": 0.80,
        "floor_abs": {
            # 在简单组里做点缀，稍微提升一点点上限
            "olympiads": 100,
            "cn_contest": 50,
            "aops_forum": 30,
        },
    },
}
'''

# =========================
# 0) 配置区 (Signal-Rich Version: Maximize Learning Signal)
# =========================

SEED = 42
SAMPLING_MODE = "bucket_source_softcap"

POOL_TOTAL = 10000 
VAL_SIZE = 128

# 保持 85/15 的配比
SUBSET_RATIO = {"default": 0.85, "extended": 0.15}

# 【改动 1：激进的 Bucket 策略】
# 优先 mixed
BUCKET_RATIO = {"mixed": 0.8, "all_correct": 0.2}

MIN_GENERATIONS = 2  
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 4096

ENFORCE_SINGLE_VALUE_ANSWER = True
CHECK_LAST_BOXED_MATCH = True

# 【改动 2：核心开关 - 解锁大量 Mixed 样本】
# 允许 dataset 自带的 generation/solution 都不对
REQUIRE_ANY_BOXED_MATCH = False  
REQUIRE_NONEMPTY_SOLUTION = True

ENABLE_DECONTAM = True
ENABLE_INTERNAL_DEDUP = True

NUM_PERM = 128
LSH_THRESHOLD = 0.8

OUTPUT_WITH_MODE_SUFFIX = True
CANONICAL_PREFIX = "math_openr1_pool10k_signal_rich"

OPENR1_SUBDIR = "OpenR1-Math/all"

# 【改动 3：Source Cap 微调】
SOURCE_SOFTCAP = {
    "default": {
        "dominant": "aops_forum", 
        "cap_frac": 0.45,       
        "floor_abs": {
            "amc_aime": 300,
            "cn_contest": 1500,    
            "olympiads": 1500,     
            "olympiads_ref": 80,
            "inequalities": 80,
            "number_theory": 50,
        },
    },
    "extended": {
        "dominant": "cn_k12",
        "cap_frac": 0.80,
        "floor_abs": {
            "olympiads": 100,
        },
    },
}


# =========================
# 1) 路径与 tokenizer（不建议改）
# =========================

def find_project_root():
    candidates = [
        "/data-store/zhaoqiannian/workspace/EGPO",
        "/data/zhaoqn/workspace/EGPO",
        os.getcwd(),
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "datasets")) and os.path.exists(os.path.join(p, "src")):
            return p
    return os.getcwd()

BASE_DIR = find_project_root()
RAW_DIR = os.path.join(BASE_DIR, "datasets", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "datasets", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OPENR1_ROOT = os.path.join(RAW_DIR, OPENR1_SUBDIR)

def list_shards(pattern: str):
    return sorted(glob.glob(os.path.join(OPENR1_ROOT, pattern)))

DEFAULT_FILES = list_shards("default-*.parquet")
EXTENDED_FILES = list_shards("extended-*.parquet")

if not DEFAULT_FILES or not EXTENDED_FILES:
    raise FileNotFoundError(
        f"OpenR1 shards not found under: {OPENR1_ROOT}\n"
        f"default={len(DEFAULT_FILES)}, extended={len(EXTENDED_FILES)}"
    )

def find_tokenizer_path():
    candidates = [
        "/data-store/zhaoqiannian/models/Qwen/Qwen3-1.7B",
        "/data/zhaoqn/models/Qwen/Qwen3-1.7B",
        "/data/zhaoqn/models/Qwen/Qwen3-8B",
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"✅ Found local tokenizer: {p}")
            return p
    print("⚠️  No local Qwen tokenizer found. Fallback to gpt2.")
    return "gpt2"

TOKENIZER_PATH = find_tokenizer_path()

def get_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")

tokenizer = get_tokenizer()


# =========================
# 2) 文本工具（boxed/归一化/多值判断）
# =========================

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

def extract_last_boxed(text: str):
    if not text:
        return None
    m = _BOXED_RE.findall(text)
    return m[-1] if m else None

def strip_thousand_commas(s: str) -> str:
    # 只移除千分位逗号：1,000,000 -> 1000000
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

def light_norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # 去掉外层 $
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = strip_thousand_commas(s)
    # 去空白
    s = re.sub(r"\s+", "", s)
    # 去掉一些 latex 间隔符
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    # 去掉最外层花括号（轻量）
    s = s.strip("{}")
    return s

def is_multi_value_answer(ans: str) -> bool:
    """
    不改 reward_manager → 必须尽量剔除多值 GT。
    这里只做“强保守”判断：出现明显分隔符/多个赋值，视为多值。
    """
    if ans is None:
        return True
    s = str(ans).strip()
    if len(s) == 0:
        return True

    s2 = strip_thousand_commas(s)
    if any(tok in s2 for tok in [";", "；"]):
        return True
    if "," in s2 or "，" in s2:
        return True

    low = s2.lower()
    if " and " in low or " or " in low or "以及" in s2:
        return True

    if s2.count("=") >= 2:
        return True

    return False

def token_len(text: str) -> int:
    try:
        return len(tokenizer.encode(str(text), add_special_tokens=False))
    except Exception:
        return len(str(text))

def to_chat_list(user_text: str, assistant_text: str):
    prompt = [{"role": "user", "content": str(user_text)}]
    response = [{"role": "assistant", "content": str(assistant_text)}]
    return prompt, response


# =========================
# 3) LSH（去污染/内部去重）
# =========================

def get_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    toks = set(re.findall(r"\w+", str(text).lower()))
    for t in toks:
        m.update(t.encode("utf-8"))
    return m

def build_decontam_index():
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    count = 0

    # 如果这些目录不存在就跳过
    test_configs = [
        ("MATH-500", ["problem", "question", "prompt"]),
        ("AIME-2024", ["problem", "question", "prompt"]),
        ("AIME-2025", ["question", "problem", "prompt"]),
        ("GPQA-Diamond", ["question", "problem", "prompt"]),
        ("big_bench_hard", ["question", "input", "prompt"]),
        ("OlympiadBench", ["question", "problem", "prompt"]),
    ]

    print("\n🔒 [Decontam] Building LSH index from test sets (if present)...")
    for name, col_cands in test_configs:
        base = os.path.join(RAW_DIR, name)
        if not os.path.exists(base):
            continue

        files = glob.glob(os.path.join(base, "**/*.parquet"), recursive=True) + \
                glob.glob(os.path.join(base, "**/*.jsonl"), recursive=True)
        if not files:
            continue

        for f in files:
            try:
                import pandas as pd
                if f.endswith(".parquet"):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_json(f, lines=True)

                col = None
                for c in col_cands:
                    if c in df.columns:
                        col = c
                        break
                if col is None:
                    continue

                for txt in df[col].dropna().astype(str).tolist():
                    if len(txt) < 16:
                        continue
                    lsh.insert(f"test_{name}_{count}", get_minhash(txt))
                    count += 1
            except Exception:
                continue

    print(f"✅ [Decontam] Index size: {count} samples.")
    return lsh

LSH_TEST = build_decontam_index() if ENABLE_DECONTAM else None


# =========================
# 4) 构造 candidates（核心：不改 schema、不依赖训练代码改动）
# =========================

def derive_seed(base_seed: int, *tokens: str) -> int:
    s = str(base_seed) + "|" + "|".join(map(str, tokens))
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def deterministic_shuffle(idxs, seed, salt):
    rng = np.random.default_rng(derive_seed(seed, salt))
    idxs = list(idxs)
    rng.shuffle(idxs)
    return idxs

def build_candidates(files, subset_name: str):
    ds = load_dataset("parquet", data_files=files, split="train")

    candidates = []
    stats = Counter()
    lsh_internal = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM) if ENABLE_INTERNAL_DEDUP else None

    for ex in tqdm(ds, desc=f"Building candidates ({subset_name})", total=len(ds)):
        problem = ex.get("problem", None)
        solution = ex.get("solution", None)
        answer = ex.get("answer", None)
        source = ex.get("source", None)
        uuid = ex.get("uuid", None)
        gens = ex.get("generations", None)
        c = ex.get("correctness_count", None)

        if not problem or not isinstance(problem, str) or len(problem.strip()) == 0:
            stats["drop_empty_problem"] += 1
            continue

        if gens is None or not isinstance(gens, list) or len(gens) < MIN_GENERATIONS:
            stats["drop_generations_short"] += 1
            continue

        if c is None:
            stats["drop_missing_correctness_count"] += 1
            continue
        try:
            c = int(c)
        except Exception:
            stats["drop_bad_correctness_count"] += 1
            continue

        N = len(gens)
        if c < 0 or c > N:
            stats["drop_inconsistent_cN"] += 1
            continue

        if answer is None or len(str(answer).strip()) == 0:
            stats["drop_empty_answer"] += 1
            continue
        if ENFORCE_SINGLE_VALUE_ANSWER and is_multi_value_answer(answer):
            stats["drop_multi_answer"] += 1
            continue

        # response 优先 solution
        response_text = solution if isinstance(solution, str) else None
        response_from = "solution"
        picked_gen_idx = -1

        if REQUIRE_NONEMPTY_SOLUTION and (response_text is None or len(response_text.strip()) == 0):
            response_text = None

        # boxed match：如果打开，就要求 solution 或某个 generation 的 last boxed 匹配 answer
        if CHECK_LAST_BOXED_MATCH:
            ans_norm = light_norm(answer)
            ok = False

            if response_text is not None:
                bx = extract_last_boxed(response_text)
                if bx is not None and light_norm(bx) == ans_norm:
                    ok = True

            if not ok:
                # fallback：找一条 generation last boxed 匹配 answer
                for i, g in enumerate(gens):
                    if not isinstance(g, str):
                        continue
                    bx = extract_last_boxed(g)
                    if bx is not None and light_norm(bx) == ans_norm:
                        response_text = g
                        response_from = "generation"
                        picked_gen_idx = i
                        ok = True
                        break

            if REQUIRE_ANY_BOXED_MATCH and not ok:
                stats["drop_no_boxed_match"] += 1
                continue

        if response_text is None:
            stats["drop_no_response"] += 1
            continue

        # token 长度过滤
        p_len = token_len(problem)
        if p_len > MAX_PROMPT_TOKENS:
            stats["drop_prompt_too_long"] += 1
            continue

        r_len = token_len(response_text)
        if r_len > MAX_RESPONSE_TOKENS:
            stats["drop_response_too_long"] += 1
            continue

        # LSH 去污染 + 内部去重（基于 problem）
        mh = get_minhash(problem)

        if ENABLE_DECONTAM and LSH_TEST is not None:
            if len(LSH_TEST.query(mh)) > 0:
                stats["drop_decontam_hit"] += 1
                continue

        if ENABLE_INTERNAL_DEDUP and lsh_internal is not None:
            if len(lsh_internal.query(mh)) > 0:
                stats["drop_internal_dup"] += 1
                continue
            lsh_internal.insert(f"{subset_name}_{uuid}_{len(candidates)}", mh)

        # bucket by (c,N)
        if c == N:
            bucket = "all_correct"
        elif c == 0:
            bucket = "all_fail"  # OpenR1-Math-220k 基本不会出现，但保留鲁棒性
        else:
            bucket = "mixed"

        prompt, response = to_chat_list(problem, response_text)

        item = {
            "data_source": "openr1_math",
            "ability": "math",
            "prompt": prompt,
            "response": response,
            "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
            "extra_info": {
                "split": "train",
                "index": -1,
                "ability": "math",
                "subset": subset_name,
                "source": str(source) if source is not None else "unknown",
                "uuid": str(uuid) if uuid is not None else "",
                "bucket": bucket,
                "correctness_count": int(c),
                "gen_count": int(N),
                "prompt_token_len": int(p_len),
                "response_token_len": int(r_len),
                "response_from": response_from,
                "picked_gen_idx": int(picked_gen_idx),
            },
        }

        candidates.append(item)
        stats[f"keep_{bucket}"] += 1

    bucket_cnt = Counter([x["extra_info"]["bucket"] for x in candidates])
    src_cnt = Counter([x["extra_info"]["source"] for x in candidates]).most_common(15)

    print(f"\n✅ Total candidates after filters/LSH ({subset_name}): {len(candidates)}")
    print(f"📊 Candidates {subset_name} bucket: {dict(bucket_cnt)}")
    print(f"📊 Candidates {subset_name} top sources: {src_cnt}")

    drops = [(k, v) for k, v in stats.items() if k.startswith("drop_")]
    drops = sorted(drops, key=lambda x: -x[1])[:25]
    if drops:
        print(f"🧹 Top drops ({subset_name}): {drops}")

    return candidates


# =========================
# 5) 采样工具
# =========================

def allocate_counts(total: int, ratios: dict, keys: list):
    w = np.array([float(ratios.get(k, 0.0)) for k in keys], dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Ratios sum to 0.")
    w = w / s

    raw = w * total
    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % len(keys)]] += 1
    return {k: int(v) for k, v in zip(keys, base)}

def pick_with_softcap(idxs, cands, subset_name, target, dominant, cap_frac, floor_abs):
    """
    在一组 idxs 上做：长尾 floors + dominant cap + 随机回填。
    idxs 是候选索引（可以是整个 subset，也可以是某个 bucket 的子集）。
    """
    if target <= 0 or not idxs:
        return []

    idxs = list(idxs)
    rng = np.random.default_rng(derive_seed(SEED, f"{subset_name}|softcap|base"))
    rng.shuffle(idxs)

    src_to = defaultdict(list)
    for i in idxs:
        src_to[cands[i]["extra_info"]["source"]].append(i)

    # 各 source 内再洗一次
    for s in list(src_to.keys()):
        src_to[s] = deterministic_shuffle(src_to[s], SEED, f"{subset_name}|softcap|src|{s}")

    picked = []
    picked_set = set()

    # 1) floors
    for s, floor in (floor_abs or {}).items():
        if len(picked) >= target:
            break
        if s not in src_to:
            continue
        take = min(int(floor), len(src_to[s]))
        take_idxs = src_to[s][:take]
        for t in take_idxs:
            if t not in picked_set:
                picked.append(t)
                picked_set.add(t)
                if len(picked) >= target:
                    break

    if len(picked) >= target:
        return picked[:target]

    # 2) 非 dominant 先填
    remaining = target - len(picked)
    non_dom = []
    for s, lst in src_to.items():
        if s == dominant:
            continue
        non_dom.extend([i for i in lst if i not in picked_set])
    non_dom = deterministic_shuffle(non_dom, SEED, f"{subset_name}|softcap|fill_non_dom")

    take = min(remaining, len(non_dom))
    picked.extend(non_dom[:take])
    picked_set.update(non_dom[:take])

    if len(picked) >= target:
        return picked[:target]

    # 3) dominant under cap
    remaining = target - len(picked)
    cap_abs = int(np.floor(cap_frac * target))
    dom_now = sum(1 for i in picked if cands[i]["extra_info"]["source"] == dominant)
    dom_slots = max(0, cap_abs - dom_now)

    dom_left = [i for i in src_to.get(dominant, []) if i not in picked_set]
    dom_left = deterministic_shuffle(dom_left, SEED, f"{subset_name}|softcap|fill_dom")

    take = min(remaining, dom_slots, len(dom_left))
    picked.extend(dom_left[:take])
    picked_set.update(dom_left[:take])

    if len(picked) >= target:
        return picked[:target]

    # 4) any fill (允许超过 cap，保证能凑满 target；softcap 是 soft）
    remaining = target - len(picked)
    any_left = []
    for s, lst in src_to.items():
        any_left.extend([i for i in lst if i not in picked_set])
    any_left = deterministic_shuffle(any_left, SEED, f"{subset_name}|softcap|fill_any")

    take = min(remaining, len(any_left))
    picked.extend(any_left[:take])

    return picked[:target]


# =========================
# 6) 三种模式的 subset 采样
# =========================

def sample_subset_pure_random(cands, subset_name, subset_target):
    idxs = deterministic_shuffle(range(len(cands)), SEED, f"{subset_name}|pure_random")
    return idxs[:subset_target]

def sample_subset_random_source_softcap(cands, subset_name, subset_target):
    cfg = SOURCE_SOFTCAP.get(subset_name, None)
    if cfg is None:
        # 没配就退化为纯随机
        return sample_subset_pure_random(cands, subset_name, subset_target)

    idxs = list(range(len(cands)))
    return pick_with_softcap(
        idxs=idxs,
        cands=cands,
        subset_name=f"{subset_name}|random_source_softcap",
        target=subset_target,
        dominant=cfg.get("dominant", ""),
        cap_frac=float(cfg.get("cap_frac", 1.0)),
        floor_abs=cfg.get("floor_abs", {}),
    )

def sample_subset_bucket_source_softcap(cands, subset_name, subset_target):
    # bucket target
    bucket_keys = list(BUCKET_RATIO.keys())
    bucket_targets = allocate_counts(subset_target, BUCKET_RATIO, bucket_keys)

    cfg = SOURCE_SOFTCAP.get(subset_name, None)

    # bucket -> idxs
    bucket_to_idxs = defaultdict(list)
    for i, it in enumerate(cands):
        bucket_to_idxs[it["extra_info"]["bucket"]].append(i)

    picked = []

    for b in bucket_keys:
        b_target = int(bucket_targets[b])
        if b_target <= 0:
            continue
        idxs_b = bucket_to_idxs.get(b, [])
        if not idxs_b:
            continue

        if cfg is None:
            # no softcap config -> random in bucket
            idxs_b = deterministic_shuffle(idxs_b, SEED, f"{subset_name}|bucket_only|{b}")
            picked.extend(idxs_b[:b_target])
        else:
            # scale floors roughly by bucket proportion（避免 floor 把小 bucket 塞爆）
            scale = b_target / max(1, subset_target)
            floor_scaled = {}
            for s, f in (cfg.get("floor_abs", {}) or {}).items():
                floor_scaled[s] = int(np.floor(f * scale))
            # 允许 floor 最少 0，避免强塞
            picked.extend(
                pick_with_softcap(
                    idxs=idxs_b,
                    cands=cands,
                    subset_name=f"{subset_name}|bucket_source_softcap|{b}",
                    target=b_target,
                    dominant=cfg.get("dominant", ""),
                    cap_frac=float(cfg.get("cap_frac", 1.0)),
                    floor_abs=floor_scaled,
                )
            )

    # backfill：凑满 subset_target（优先从 bucket_keys 内剩余，再从所有剩余）
    if len(picked) < subset_target:
        picked_set = set(picked)
        remaining_all = [i for i in range(len(cands)) if i not in picked_set]
        remaining_all = deterministic_shuffle(remaining_all, SEED, f"{subset_name}|bucket_source_softcap|backfill|any")
        need = subset_target - len(picked)
        picked.extend(remaining_all[:need])

    return picked[:subset_target]

def sample_subset_dispatch(cands, subset_name, subset_target):
    if SAMPLING_MODE == "pure_random":
        return sample_subset_pure_random(cands, subset_name, subset_target)
    if SAMPLING_MODE == "random_source_softcap":
        return sample_subset_random_source_softcap(cands, subset_name, subset_target)
    if SAMPLING_MODE == "bucket_source_softcap":
        return sample_subset_bucket_source_softcap(cands, subset_name, subset_target)
    raise ValueError(f"Unknown SAMPLING_MODE={SAMPLING_MODE}")


# =========================
# 7) 保存（核武器级清洗，避免 numpy/arrow 污染）
# =========================

def clean_obj_recursive(obj):
    if isinstance(obj, np.ndarray):
        return [clean_obj_recursive(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [clean_obj_recursive(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean_obj_recursive(v) for k, v in obj.items()}
    return obj

def save_safe_parquet(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cleaned = [clean_obj_recursive(r) for r in rows]
    ds = Dataset.from_list(cleaned)
    ds.to_parquet(out_path)
    print(f"✅ Saved: {out_path} | rows={len(ds)}")
    return ds


# =========================
# 8) 主流程
# =========================

def main():
    print("=" * 78)
    print("🚀 OpenR1-Math Pool-30k Builder (EGPO, RL+SFT compatible)")
    print("=" * 78)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OPENR1_ROOT: {OPENR1_ROOT}")
    print(f"SAMPLING_MODE: {SAMPLING_MODE}")
    print(f"SEED: {SEED}")
    print(f"POOL_TOTAL: {POOL_TOTAL} | VAL_SIZE: {VAL_SIZE}")
    print(f"SUBSET_RATIO: {SUBSET_RATIO}")
    print(f"BUCKET_RATIO: {BUCKET_RATIO}")
    print(f"CHECK_LAST_BOXED_MATCH: {CHECK_LAST_BOXED_MATCH} | REQUIRE_ANY_BOXED_MATCH: {REQUIRE_ANY_BOXED_MATCH}")
    print(f"DECONTAM: {ENABLE_DECONTAM} | INTERNAL_DEDUP: {ENABLE_INTERNAL_DEDUP}")
    if SAMPLING_MODE in ["random_source_softcap", "bucket_source_softcap"]:
        print(f"SOURCE_SOFTCAP: {json.dumps(SOURCE_SOFTCAP, ensure_ascii=False)}")
    print("-" * 78)

    # 1) candidates
    cands_default = build_candidates(DEFAULT_FILES, "default")
    cands_extended = build_candidates(EXTENDED_FILES, "extended")
    total_candidates = len(cands_default) + len(cands_extended)
    print(f"\n✅ Total candidates after filters/LSH: {total_candidates}")

    # 2) subset targets
    subset_targets = allocate_counts(POOL_TOTAL, SUBSET_RATIO, ["default", "extended"])
    tgt_default = subset_targets["default"]
    tgt_extended = subset_targets["extended"]
    print(f"\n🎯 Pool targets: default={tgt_default}, extended={tgt_extended}, total={POOL_TOTAL}")

    if len(cands_default) < tgt_default:
        raise RuntimeError(f"default candidates too few: {len(cands_default)} < target {tgt_default}")
    if len(cands_extended) < tgt_extended:
        raise RuntimeError(f"extended candidates too few: {len(cands_extended)} < target {tgt_extended}")

    # 3) sample
    pick_default = sample_subset_dispatch(cands_default, "default", tgt_default)
    pick_extended = sample_subset_dispatch(cands_extended, "extended", tgt_extended)

    pool = [cands_default[i] for i in pick_default] + [cands_extended[i] for i in pick_extended]

    # deterministic shuffle pool
    rng = np.random.default_rng(derive_seed(SEED, "pool_shuffle"))
    rng.shuffle(pool)

    # fill index/split
    for i, r in enumerate(pool):
        r["extra_info"]["index"] = int(i)
        r["extra_info"]["split"] = "pool"

    # pool stats
    bucket_cnt = Counter([x["extra_info"]["bucket"] for x in pool])
    src_cnt = Counter([x["extra_info"]["source"] for x in pool]).most_common(20)
    sub_cnt = Counter([x["extra_info"]["subset"] for x in pool])

    print(f"\n✅ Pool selected: {len(pool)} rows")
    print(f"📊 Pool subset dist: {dict(sub_cnt)}")
    print(f"📊 Pool bucket dist: {dict(bucket_cnt)}")
    print(f"📊 Pool top sources: {src_cnt}")

    # 4) split train/val
    assert VAL_SIZE < len(pool), "VAL_SIZE must be smaller than pool size."
    rng2 = np.random.default_rng(derive_seed(SEED, "val_split"))
    idxs = np.arange(len(pool))
    rng2.shuffle(idxs)

    val_idxs = set(idxs[:VAL_SIZE].tolist())
    val_rows, train_rows = [], []

    for i, r in enumerate(pool):
        if i in val_idxs:
            r["extra_info"]["split"] = "val"
            val_rows.append(r)
        else:
            r["extra_info"]["split"] = "train"
            train_rows.append(r)

    # 5) names
    prefix = CANONICAL_PREFIX
    if OUTPUT_WITH_MODE_SUFFIX:
        prefix = f"{CANONICAL_PREFIX}_{SAMPLING_MODE}"

    recipe_path = os.path.join(PROCESSED_DIR, f"{prefix}_recipe.json")
    uuid_path = os.path.join(PROCESSED_DIR, f"{prefix}_uuid_list.txt")

    recipe = {
        "seed": SEED,
        "sampling_mode": SAMPLING_MODE,
        "pool_total": POOL_TOTAL,
        "val_size": VAL_SIZE,
        "subset_ratio": SUBSET_RATIO,
        "bucket_ratio": BUCKET_RATIO,
        "source_softcap": SOURCE_SOFTCAP if SAMPLING_MODE in ["random_source_softcap", "bucket_source_softcap"] else None,
        "filters": {
            "min_generations": MIN_GENERATIONS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "enforce_single_value_answer": ENFORCE_SINGLE_VALUE_ANSWER,
            "check_last_boxed_match": CHECK_LAST_BOXED_MATCH,
            "require_any_boxed_match": REQUIRE_ANY_BOXED_MATCH,
            "require_nonempty_solution": REQUIRE_NONEMPTY_SOLUTION,
        },
        "lsh": {
            "enable_decontam": ENABLE_DECONTAM,
            "enable_internal_dedup": ENABLE_INTERNAL_DEDUP,
            "num_perm": NUM_PERM,
            "threshold": LSH_THRESHOLD,
        },
        "candidate_stats": {
            "default": {
                "count": len(cands_default),
                "bucket": dict(Counter([x["extra_info"]["bucket"] for x in cands_default])),
                "top_sources": Counter([x["extra_info"]["source"] for x in cands_default]).most_common(20),
            },
            "extended": {
                "count": len(cands_extended),
                "bucket": dict(Counter([x["extra_info"]["bucket"] for x in cands_extended])),
                "top_sources": Counter([x["extra_info"]["source"] for x in cands_extended]).most_common(20),
            },
        },
        "pool_stats": {
            "subset": dict(sub_cnt),
            "bucket": dict(bucket_cnt),
            "top_sources": src_cnt,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "paths": {"base_dir": BASE_DIR, "openr1_root": OPENR1_ROOT},
    }

    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    with open(uuid_path, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(str(r["extra_info"].get("uuid", "")) + "\n")

    print(f"\n🧾 Saved recipe: {recipe_path}")
    print(f"🧾 Saved uuid list: {uuid_path}")

    # 6) save parquet
    train_out = os.path.join(PROCESSED_DIR, f"{prefix}_train_final.parquet")
    val_out = os.path.join(PROCESSED_DIR, f"{prefix}_val_fixed.parquet")

    print("\n💾 Saving final parquet files (safe-clean)...\n")
    save_safe_parquet(train_rows, train_out)
    save_safe_parquet(val_rows, val_out)

    print("\n🎉 DONE. Files are ready for EGPO training.")
    print(f"   Train: {train_out}")
    print(f"   Val  : {val_out}")

    if ALSO_WRITE_CANONICAL_COPY:
        canon_train = os.path.join(PROCESSED_DIR, f"{CANONICAL_PREFIX}_train_final.parquet")
        canon_val = os.path.join(PROCESSED_DIR, f"{CANONICAL_PREFIX}_val_fixed.parquet")
        if canon_train != train_out:
            print("\n📌 Also writing canonical copies (no mode suffix)...\n")
            save_safe_parquet(train_rows, canon_train)
            save_safe_parquet(val_rows, canon_val)
            print(f"   Canon Train: {canon_train}")
            print(f"   Canon Val  : {canon_val}")

if __name__ == "__main__":
    main()
