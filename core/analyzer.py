"""
TensorBoard 日志读取与分析模块。
读取训练 event 文件，提取标量数据、配置信息，计算训练诊断指标。
"""

import os
import pickle
import glob
import numpy as np
from dataclasses import dataclass, field
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class RunData:
    """单次训练运行的数据。"""
    name: str
    log_dir: str
    scalars: dict = field(default_factory=dict)  # key -> {"steps": [], "values": []}
    config: dict = field(default_factory=dict)    # cfgs.pkl 中的配置
    total_iters: int = 0

    @property
    def reward_values(self):
        return self.scalars.get("Train/mean_reward", {}).get("values", [])

    @property
    def reward_steps(self):
        return self.scalars.get("Train/mean_reward", {}).get("steps", [])


def find_event_file(log_dir):
    """查找 TensorBoard event 文件。"""
    for root, _dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return os.path.join(root, f)
    return None


def load_scalars(log_dir):
    """从日志目录读取所有标量数据。"""
    event_file = find_event_file(log_dir)
    if not event_file:
        return {}

    acc = EventAccumulator(event_file)
    acc.Reload()

    scalars = {}
    for key in acc.scalars.Keys():
        items = acc.scalars.Items(key)
        scalars[key] = {
            "steps": [e.step for e in items],
            "values": [e.value for e in items],
        }
    return scalars


def load_config(log_dir):
    """从 cfgs.pkl 读取训练配置。"""
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "rb") as f:
            data = pickle.load(f)
        # data = [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg]
        keys = ["env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "train_cfg"]
        return dict(zip(keys, data))
    except Exception:
        return {}


def scan_log_dirs(base_dir="logs"):
    """扫描日志目录，返回所有训练运行列表。"""
    if not os.path.exists(base_dir):
        return []

    runs = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        # 检查是否有 event 文件或 cfgs.pkl
        if find_event_file(path) or os.path.exists(os.path.join(path, "cfgs.pkl")):
            runs.append(path)
    return runs


def load_run(log_dir):
    """加载单次训练的完整数据。"""
    name = os.path.basename(log_dir.rstrip("/"))
    scalars = load_scalars(log_dir)
    config = load_config(log_dir)

    total_iters = 0
    if "Train/mean_reward" in scalars:
        total_iters = len(scalars["Train/mean_reward"]["steps"])

    return RunData(
        name=name,
        log_dir=log_dir,
        scalars=scalars,
        config=config,
        total_iters=total_iters,
    )


def compute_diagnostics(run: RunData):
    """分析训练数据，返回诊断结果和建议。"""
    issues = []
    suggestions = []

    reward = run.reward_values
    if not reward:
        return {"issues": ["无 reward 数据"], "suggestions": [], "summary": "数据不足"}

    reward = np.array(reward)
    final_avg = np.mean(reward[-100:]) if len(reward) > 100 else np.mean(reward)
    peak = np.max(reward)
    final_10 = np.mean(reward[-10:]) if len(reward) >= 10 else final_avg

    # 1. Reward 不增长或下降
    if len(reward) > 200:
        first_half = np.mean(reward[:len(reward)//2])
        second_half = np.mean(reward[len(reward)//2:])
        if second_half < first_half * 0.9:
            issues.append("Reward 后半程下降，训练可能不稳定")
            suggestions.append("降低学习率（如 0.0003 → 0.0001）")
            suggestions.append("增大 entropy 鼓励探索（如 0.002 → 0.005）")

    # 2. Reward 平坦不增长
    if len(reward) > 100:
        std_last = np.std(reward[-100:])
        mean_last = np.mean(reward[-100:])
        if mean_last > 0 and std_last / mean_last < 0.05:
            # 变异系数很小，可能收敛了也可能卡住了
            if final_avg < peak * 0.8:
                issues.append("Reward 波动小但低于峰值，可能陷入局部最优")
                suggestions.append("增大 init_noise_std 增加探索（如 1.0 → 1.5）")
                suggestions.append("尝试增大网络容量（如 [128,128] → [256,256]）")

    # 3. 碰撞率高
    crash = run.scalars.get("Episode/rew_crash", {}).get("values", [])
    if crash:
        crash_arr = np.array(crash)
        final_crash = np.mean(crash_arr[-50:]) if len(crash_arr) > 50 else np.mean(crash_arr)
        if abs(final_crash) > 0.5:
            issues.append(f"碰撞惩罚高 ({final_crash:.4f})，坠机频繁")
            suggestions.append("增大 crash 惩罚权重（如 -10 → -20）")
            suggestions.append("检查 max_safety_violations 是否过低")

    # 4. 安全约束惩罚高（通用检测：任何 safety 相关的 reward 项）
    for safety_key in ["Episode/rew_cable_angle_safety", "Episode/rew_safety", "Episode/rew_constraint"]:
        safety = run.scalars.get(safety_key, {}).get("values", [])
        if safety:
            safety_arr = np.array(safety)
            final_safety = np.mean(safety_arr[-50:]) if len(safety_arr) > 50 else np.mean(safety_arr)
            if abs(final_safety) > 0.01:
                issues.append(f"安全约束惩罚高 {safety_key} ({final_safety:.4f})")
                suggestions.append("检查安全约束阈值是否过紧，适当放宽")
                suggestions.append("适当降低该惩罚项的权重")

    # 5. 动作噪声不收敛
    noise = run.scalars.get("Policy/mean_noise_std", {}).get("values", [])
    if noise and len(noise) > 50:
        noise_arr = np.array(noise)
        if noise_arr[-1] > noise_arr[0] * 0.8:
            issues.append("动作噪声未明显下降，策略未充分收敛")
            suggestions.append("增加训练迭代次数")
            suggestions.append("检查学习率是否过小")

    # 6. Value loss 不下降
    vloss = run.scalars.get("Loss/value_function", {}).get("values", [])
    if vloss and len(vloss) > 200:
        vloss_arr = np.array(vloss)
        first_vl = np.mean(vloss_arr[:50])
        last_vl = np.mean(vloss_arr[-50:])
        if last_vl > first_vl * 0.9:
            issues.append("Value function loss 未下降，Critic 学习不佳")
            suggestions.append("增大 Critic 网络容量")
            suggestions.append("检查奖励尺度是否合理（reward 过大会导致 value loss 震荡）")

    if not issues:
        summary = f"训练状态良好。最终 reward: {final_avg:.4f}，峰值: {peak:.4f}"
    else:
        summary = f"发现 {len(issues)} 个潜在问题"

    return {
        "issues": issues,
        "suggestions": suggestions,
        "summary": summary,
        "metrics": {
            "final_reward": float(final_avg),
            "peak_reward": float(peak),
            "final_reward_10": float(final_10),
        },
    }
