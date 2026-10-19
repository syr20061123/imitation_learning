import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t
from scipy.spatial.distance import cdist
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "q6_outputs"
SEEDS = range(8)
N_DEMOS, ROUNDS, HORIZON = 3, 3, 200
N_TEST, N_PROBE = 100, 100
N_CANDIDATE_TRAJS, NOVELTY_K = 3, 5
# 固定物理尺度和时间尺度，不从测试集或未查询专家标签估计。
NOVELTY_SCALE = np.array([1.0, 0.5, 2.0])
TRAIN_STEPS, BATCH_SIZE, LR = 1500, 256, 1e-3
ACTION_SCALE = 5.0
GROUPS = ("BC", "MORE_EXPERT", "RANDOM_DAGGER", "DAGGER")
NAMES = {"BC": "初始行为克隆", "MORE_EXPERT": "普通专家扩充",
         "RANDOM_DAGGER": "随机状态纠偏", "DAGGER": "关键状态纠偏"}

# 状态为位置、速度，动作为加速度。所有组共用无扰动、无裁剪的线性动力学。
A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q = np.diag([1.0, 0.1])
R = np.array([[0.02]])
Q_FINAL = np.diag([5.0, 1.0])


def rng_for(seed, role):
    return np.random.default_rng(np.random.SeedSequence([20261017, seed, role]))


def initial_states(seed, role, count):
    # 所有采集和测试共用此分布，随机流分离；四个实验组的测试初态配对。
    return rng_for(seed, role).uniform(-1, 1, (count, 2)) * [1.0, 0.5]


class OptimalExpert:
    def __init__(self):
        self.training_queries = 0
        self.p = np.empty((HORIZON + 1, 2, 2))
        self.k = np.empty((HORIZON, 1, 2))
        self.s = np.empty(HORIZON)
        self.p[-1] = Q_FINAL
        for step in reversed(range(HORIZON)):
            p_next = self.p[step + 1]
            hessian = R + B.T @ p_next @ B
            gain = np.linalg.solve(hessian, B.T @ p_next @ A)
            self.k[step] = gain
            self.s[step] = hessian.item()
            p = Q + A.T @ p_next @ A - A.T @ p_next @ B @ gain
            self.p[step] = (p + p.T) / 2

    def action(self, state, step):
        return -np.einsum("...i,...i->...", state, self.k[step, 0])[:, None]

    def cost(self, initial):
        return np.einsum("bi,ij,bj->b", initial, self.p[0], initial)


def features(state, step):
    # 有限时域最优策略依赖剩余时间；模型只获得当前状态与时间，不获得未来真值。
    remaining = np.full((len(state), 1), 1 - 2 * step / (HORIZON - 1))
    return np.column_stack((state, remaining))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, observations):
    x = torch.as_tensor(observations, dtype=torch.float32)
    return model(x).numpy().astype(np.float64) * ACTION_SCALE


def fit(observations, labels, seed):
    # 统一初始化、优化器、批量、梯度更新数；不按数据量改变 epoch 或早停。
    torch.manual_seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed + 91001)
    x = torch.tensor(observations, dtype=torch.float32)
    y = torch.tensor(labels / ACTION_SCALE, dtype=torch.float32)
    for _ in range(TRAIN_STEPS):
        indices = torch.randint(len(x), (BATCH_SIZE,), generator=generator)
        loss = nn.functional.mse_loss(model(x[indices]), y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    train_mse = float(np.mean((predict(model, observations) - labels) ** 2))
    if not np.isfinite(train_mse):
        raise RuntimeError("训练出现非有限数值；停止并报告，不删除失败种子。")
    return model, train_mse


def rollout(initial, action_fn):
    """仅执行给定策略；学习者候选采集不会访问专家。返回执行动作，不是专家标签。"""
    state = initial.copy()
    observations = np.empty((len(state), HORIZON, 3))
    actions = np.empty((len(state), HORIZON, 1))
    cost = np.zeros(len(state))
    for step in range(HORIZON):
        obs = features(state, step)
        action = np.asarray(action_fn(obs, step))
        observations[:, step], actions[:, step] = obs, action
        cost += np.einsum("bi,ij,bj->b", state, Q, state) + R.item() * action[:, 0] ** 2
        state = state @ A.T + action @ B.T
    cost += np.einsum("bi,ij,bj->b", state, Q_FINAL, state)
    if not np.isfinite(cost).all():
        raise RuntimeError("轨迹出现非有限代价；不截断或丢弃失败轨迹。")
    return observations, actions, cost


def expert_targets(expert, observations):
    steps = np.rint((1 - observations[:, 2]) * (HORIZON - 1) / 2).astype(int)
    assert np.all((0 <= steps) & (steps < HORIZON))
    return expert.action(observations[:, :2], steps)


def query_training_labels(expert, observations):
    expert.training_queries += len(observations)
    return expert_targets(expert, observations)


def collect_expert(expert, initial):
    return rollout(initial, lambda obs, step: query_training_labels(expert, obs))[:2]


def verify_cost_identity(expert, initial, data):
    observations, actions, cost = data
    targets = expert_targets(expert, observations.reshape(-1, 3)).reshape(actions.shape)
    advantage = np.sum(expert.s[None, :] * (actions[:, :, 0] - targets[:, :, 0]) ** 2, axis=1)
    # 仅用于独立验证，不在未标注候选的生成和筛选中调用。
    optimal = expert.cost(initial)
    np.testing.assert_allclose(cost - optimal, advantage, rtol=1e-8, atol=1e-8)


def verify_expert(expert):
    # 确定性、全状态最优依赖 Q,Qf 半正定、R正定及无约束；不是训练出的近似专家。
    assert np.linalg.eigvalsh(Q).min() >= 0 and np.linalg.eigvalsh(Q_FINAL).min() >= 0
    assert np.linalg.eigvalsh(R).min() > 0
    initial = initial_states(91000, 0, 12)
    data = rollout(initial, lambda obs, step: expert.action(obs[:, :2], step))
    cost = data[2]
    np.testing.assert_allclose(cost, expert.cost(initial), rtol=1e-10, atol=1e-10)
    verify_cost_identity(expert, initial, data)
    # 非零扰动验证 J(pi)-J* = sum S_t (u_t-u*_t)^2，避免仅检查0=0。
    perturbed = lambda obs, step: expert.action(obs[:, :2], step) + 0.15 * np.cos(step)
    changed = rollout(initial, perturbed)
    verify_cost_identity(expert, initial, changed)
    assert np.all(changed[2] > cost)
    assert expert.training_queries == 0


def flatten(data):
    return data[0].reshape(-1, 3), data[1].reshape(-1, 1)


def novelty_scores(candidates, reference):
    shape = candidates.shape[:2]
    distances = cdist(candidates.reshape(-1, 3) / NOVELTY_SCALE,
                      reference / NOVELTY_SCALE)
    k = min(NOVELTY_K, len(reference))
    scores = np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1)
    return scores.reshape(shape)


def select_corrections(candidates, reference, seed, round_id, active):
    # 每个时刻选1个状态：随机组与关键组候选数、标签数、时间覆盖相同。
    scores = novelty_scores(candidates, reference)
    if active:
        episodes = scores.argmax(axis=0)
    else:
        episodes = rng_for(seed, 200 + round_id).integers(len(candidates), size=HORIZON)
    times = np.arange(HORIZON)
    selected = candidates[episodes, times].copy()
    np.testing.assert_allclose(selected[:, 2], 1 - 2 * times / (HORIZON - 1))
    return selected, float(scores[episodes, times].mean()), float(scores.mean())


def build_groups(expert, seed):
    query_start = expert.training_queries
    base = flatten(collect_expert(expert, initial_states(seed, 0, N_DEMOS)))
    n0 = N_DEMOS * HORIZON
    assert expert.training_queries - query_start == n0
    model_seed = 120000 + seed
    initial_bc, initial_train_mse = fit(*base, seed=model_seed)
    groups = {group: tuple(a.copy() for a in base) for group in GROUPS}
    models = {group: initial_bc for group in GROUPS}
    train_mses = {group: initial_train_mse for group in GROUPS}
    audits = {group: dict(added_expert_queries=0, added_env_steps=0,
                         selected_novelty=[], candidate_novelty=[]) for group in GROUPS}
    for round_id in range(1, ROUNDS + 1):
        initial = initial_states(seed, round_id, N_CANDIDATE_TRAJS)
        for group in GROUPS[1:]:
            before = expert.training_queries
            if group == "MORE_EXPERT":
                # 不根据质量挑选专家轨迹；固定取首个初态，3轮新增3条完整示范。
                added = flatten(collect_expert(expert, initial[:1]))
                audits[group]["added_env_steps"] += HORIZON
            else:
                candidate_data = rollout(initial, lambda obs, step: predict(models[group], obs))
                candidates = candidate_data[0]
                assert expert.training_queries == before
                chosen, selected_score, candidate_score = select_corrections(
                    candidates, groups[group][0], seed, round_id, active=group == "DAGGER")
                # 到此才查询这200个状态。未选候选不查询、不训练，不执行专家接管。
                assert expert.training_queries == before
                added = chosen, query_training_labels(expert, chosen)
                audits[group]["added_env_steps"] += len(initial) * HORIZON
                audits[group]["selected_novelty"].append(selected_score)
                audits[group]["candidate_novelty"].append(candidate_score)
            queried = expert.training_queries - before
            assert queried == len(added[0]) == HORIZON
            audits[group]["added_expert_queries"] += queried
            groups[group] = tuple(np.concatenate((old, new)) for old, new in zip(groups[group], added))
            # 三个新增组每轮都重置到同一初始化，采用相同拟合预算。
            models[group], train_mses[group] = fit(*groups[group], seed=model_seed)
    for group in GROUPS[1:]:
        assert len(groups[group][0]) == n0 + ROUNDS * HORIZON
        assert audits[group]["added_expert_queries"] == ROUNDS * HORIZON
        for original, expanded in zip(groups["BC"], groups[group]):
            np.testing.assert_array_equal(original, expanded[:n0])
        expected_times = np.repeat(np.arange(HORIZON), N_DEMOS + ROUNDS)
        actual_times = np.rint((1 - groups[group][0][:, 2]) * (HORIZON - 1) / 2).astype(int)
        np.testing.assert_array_equal(np.sort(actual_times), expected_times)
    assert expert.training_queries - query_start == n0 + 3 * ROUNDS * HORIZON
    return groups, models, train_mses, audits


def run_experiment():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    expert = OptimalExpert()
    verify_expert(expert)
    rows = []
    for seed in SEEDS:
        groups, models, train_mses, audits = build_groups(expert, seed)
        test_initial = initial_states(seed, 100, N_TEST)
        optimal = expert.cost(test_initial).mean()
        # 使用同一初始BC的独立测试轨迹作为固定纠偏探针；各组不在不同状态上比较MSE。
        probe = rollout(initial_states(seed, 101, N_PROBE),
                        lambda obs, step: predict(models["BC"], obs))
        probe_x = probe[0].reshape(-1, 3)
        probe_y = expert_targets(expert, probe_x)
        n0 = N_DEMOS * HORIZON
        for group in GROUPS:
            observations, labels = groups[group]
            model = models[group]
            data = rollout(test_initial, lambda obs, step: predict(model, obs))
            verify_cost_identity(expert, test_initial, data)
            costs = data[2]
            audit = audits[group]
            new_expert_count = ROUNDS * HORIZON if group == "MORE_EXPERT" else 0
            correction_count = len(labels) - n0 - new_expert_count
            rows.append(dict(
                seed=seed, group=group, initial_demo_trajectories=N_DEMOS, horizon=HORIZON,
                train_samples=len(labels), initial_samples_retained=n0, new_expert_samples=new_expert_count,
                correction_samples=correction_count, final_fit_updates=TRAIN_STEPS,
                added_expert_queries=audit["added_expert_queries"],
                added_env_steps=audit["added_env_steps"],
                total_fit_updates=TRAIN_STEPS * (1 if group == "BC" else ROUNDS + 1),
                selected_novelty_mean=(float(np.mean(audit["selected_novelty"]))
                                       if audit["selected_novelty"] else ""),
                candidate_novelty_mean=(float(np.mean(audit["candidate_novelty"]))
                                        if audit["candidate_novelty"] else ""),
                test_cost=float(costs.mean()), optimal_cost=float(optimal),
                excess_cost_pct=float(100 * (costs.mean() - optimal) / optimal),
                probe_action_mse=float(np.mean((predict(model, probe_x) - probe_y) ** 2)),
                train_action_mse=train_mses[group],
            ))
        print(f"种子 {seed + 1}/{len(SEEDS)} 完成", flush=True)
    return rows


def values(rows, group, metric="excess_cost_pct"):
    selected = sorted((r for r in rows if r["group"] == group), key=lambda r: r["seed"])
    return np.array([r[metric] for r in selected])


def contrasts(rows):
    base, more, random, dagger = [values(rows, group) for group in GROUPS]
    # 核心 more-dagger 比较等标签预算；random-dagger 检验筛选机制的整体额外作用。
    planned = [
        ("普通扩充收益：初始－普通扩充", base - more),
        ("随机纠偏收益：普通扩充－随机纠偏", more - random),
        ("关键筛选收益：随机纠偏－关键纠偏", random - dagger),
        ("纠偏总额外收益：普通扩充－关键纠偏", more - dagger),
    ]
    np.testing.assert_allclose(base - dagger, (base - more) + (more - random) + (random - dagger))
    # 指定比较路径上的算术分解，不意味着两个机制独立或可给出唯一贡献百分比。
    # 以8个配对种子为统计单位；四项预设比较用Bonferroni校正近似95%同时t区间。
    # 区间包含0仅表示证据不足，不能当作两组等效或“只有数量起作用”的证明。
    critical = student_t.ppf(1 - 0.05 / (2 * len(planned)), len(base) - 1)
    return [(label, diff.mean(), critical * diff.std(ddof=1) / np.sqrt(len(diff)))
            for label, diff in planned]


def plot_results(rows, effects):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    data = [values(rows, group) for group in GROUPS]
    counts = [next(r["train_samples"] for r in rows if r["group"] == group) for group in GROUPS]
    colors = ["tab:blue", "tab:green", "gray", "tab:orange"]
    axes[0].bar(np.arange(len(GROUPS)), [d.mean() for d in data],
                yerr=[d.std(ddof=1) for d in data], capsize=4, color=colors, alpha=0.75)
    ticklabels = [f"{NAMES[group]}\n{count}个标注对" for group, count in zip(GROUPS, counts)]
    axes[0].set(xticks=np.arange(len(GROUPS)), xticklabels=ticklabels,
                ylabel="相对最优策略的额外代价（%）", title="少示范消融：均值 ± 1个标准差")
    axes[0].tick_params(axis="x", labelsize=9)
    labels, means, widths = zip(*effects)
    for i, (mean, width) in enumerate(zip(means, widths)):
        axes[1].errorbar(mean, i, xerr=width, fmt="o", capsize=4,
                        color="tab:orange" if i == 3 else "tab:blue")
    axes[1].set(yticks=np.arange(len(effects)), yticklabels=labels,
                xlabel="额外代价的降低量（百分点）", title="配对归因：近似95%同时置信区间")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].tick_params(axis="y", labelsize=9)
    for ax in axes:
        ax.grid(axis="y" if ax == axes[0] else "x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{N_DEMOS}条共同初始示范｜普通专家扩充至{N_DEMOS + ROUNDS}条｜"
                 f"纠偏新增{ROUNDS * HORIZON}个标签｜{len(SEEDS)}个配对种子")
    fig.tight_layout()
    fig.savefig(OUT / "q6_ablation.png", dpi=160)
    plt.close(fig)


def main():
    rows = run_experiment()
    OUT.mkdir(exist_ok=True)
    with (OUT / "q6_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effects = contrasts(rows)
    plot_results(rows, effects)
    print(f"\n少示范消融（{len(SEEDS)}个种子的均值±标准差；越低越好）")
    for group in GROUPS:
        cost = values(rows, group)
        count = next(r["train_samples"] for r in rows if r["group"] == group)
        print(f"{NAMES[group]}（{count}个标注对）：额外代价 {cost.mean():.3f}% ± {cost.std(ddof=1):.3f}%")
    print("\n配对归因及Bonferroni校正近似95%同时置信区间（单位：百分点）")
    for label, mean, width in effects:
        print(f"{label}：{mean:.3f} [{mean - width:.3f}, {mean + width:.3f}]")
    print("关键状态按已有数据的近邻距离筛选；随机纠偏对照用于检验筛选是否有额外收益。")
    print(f"新增专家标签均{ROUNDS * HORIZON}个；新增交互：普通{ROUNDS * HORIZON}步，"
          f"两纠偏组各{ROUNDS * N_CANDIDATE_TRAJS * HORIZON}步。")

if __name__ == "__main__":
    main()