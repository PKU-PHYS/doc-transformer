"""训练入口 — 支持表格数据的课程学习训练。

  Stage 0: 少行预热 (n_rows=2-3)
  Stage 1: 中等行数 (n_rows=5-8)
  Stage 2: 多行复杂推断 (n_rows=15-30)

特性：
  - 基于 loss 收敛自动切换阶段 (patience-based)
  - Loss 曲线自动保存为 PNG 图片
  - 定期保存 checkpoint (每 N epochs)
  - 支持 --resume 断点续训，含完整 RNG 状态恢复
"""

import os
import time
import json
import random
import argparse
import functools
import torch
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig, TrainConfig
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from data.base import collate_fn
from data.tabular import TabularDataset, TableLoader
from eval import evaluate
# Matbench imports are deferred to main() to avoid import overhead


SEED = 42


def set_seed(seed: int):
    """设置所有随机源的种子，确保完全可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 cuDNN 使用确定性算法（会略微降速 ~5%）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_states() -> dict:
    """捕获所有随机源的状态，用于 checkpoint 保存。"""
    states = {
        "python_rng": random.getstate(),
        "torch_rng": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict):
    """从 checkpoint 恢复所有随机源的状态。"""
    if "python_rng" in states:
        random.setstate(states["python_rng"])
    if "torch_rng" in states:
        # set_rng_state 要求 CPU ByteTensor；checkpoint 可能被 map_location 移到了 GPU
        torch.random.set_rng_state(states["torch_rng"].cpu().byte())
    if "cuda_rng" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().byte() for s in states["cuda_rng"]])


def plot_loss_curves(all_logs: dict, save_dir: str):
    """绘制所有 Stage 的 loss 曲线并保存为 PNG。"""
    os.makedirs(save_dir, exist_ok=True)

    # ── 分阶段子图 ──
    fig, axes = plt.subplots(1, len(all_logs), figsize=(6 * len(all_logs), 5))
    if len(all_logs) == 1:
        axes = [axes]

    _PALETTE = ["#9C27B0", "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#00BCD4"]

    for idx, (ax, (stage_key, log)) in enumerate(zip(axes, all_logs.items())):
        losses = log["losses"]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(range(1, len(losses) + 1), losses, color=color, linewidth=1.5)
        ax.set_title(f'{log["stage"]}', fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Loss")
        ax.grid(True, alpha=0.3)

        # 标注最低 loss
        if losses:
            min_loss = min(losses)
            min_epoch = losses.index(min_loss) + 1
            ax.annotate(f"min={min_loss:.4f}\nepoch {min_epoch}",
                        xy=(min_epoch, min_loss),
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->", color=color),
                        xytext=(min_epoch + len(losses) * 0.1, min_loss * 1.2))

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_per_stage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Loss per stage plot: {path}")

    # ── 全局合并图 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    global_epoch = 0
    stage_boundaries = []

    for idx, (stage_key, log) in enumerate(all_logs.items()):
        losses = log["losses"]
        color = _PALETTE[idx % len(_PALETTE)]
        epochs = list(range(global_epoch + 1, global_epoch + len(losses) + 1))
        ax.plot(epochs, losses, color=color, linewidth=1.5, label=log["stage"])
        stage_boundaries.append(global_epoch)
        global_epoch += len(losses)

    # 阶段分界线
    for boundary in stage_boundaries[1:]:
        ax.axvline(x=boundary + 0.5, color="#999", linestyle="--", alpha=0.5)

    ax.set_title("Training Loss — All Stages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Avg Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "loss_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Global loss plot: {path}")


@torch.no_grad()
def _log_sample_case(model, out, batched_leaves, batched_masks,
                     sample_idx=0, stage_name="", epoch=0, batch_idx=0):
    """诊断输出：展示一个样本的输入、mask、预测值和真实值。"""
    if sample_idx >= len(batched_leaves):
        return

    leaves = batched_leaves[sample_idx]
    masks = batched_masks[sample_idx]

    if not leaves or not masks:
        return

    print(f"\n  {'─'*60}")
    print(f"  🔍 Sample Case [{stage_name}] E{epoch:03d} B{batch_idx:04d}")
    print(f"  {'─'*60}")

    def _format_leaf(i, leaf, is_masked):
        path_str = ".".join(leaf.path)
        marker = "  🎯" if is_masked else ""
        if leaf.value_type == "number":
            val_str = f"{leaf.value:.4g}"
        elif leaf.value_type == "string":
            val_str = f'"{leaf.value[:20]}"' if len(str(leaf.value)) > 20 else f'"{leaf.value}"'
        else:
            val_str = str(leaf.value)
        return f"    [{i:3d}] {path_str:40s} = {val_str:15s} ({leaf.value_type}){marker}"

    # 按 group_ids 分行显示
    print(f"  Leaves ({len(leaves)} tokens):")
    current_group = None
    row_num = -1
    display_limit = min(len(leaves), 30)
    for i in range(display_limit):
        leaf = leaves[i]
        leaf_group = tuple(leaf.group_ids) if leaf.group_ids else ()
        if leaf_group != current_group:
            current_group = leaf_group
            row_num += 1
            print(f"    ── Row {row_num} ──")
        print(_format_leaf(i, leaf, i in masks))
    if len(leaves) > display_limit:
        print(f"    ... ({len(leaves) - display_limit} more tokens)")

    # 推断上下文
    def _find_context(target_leaf, target_idx):
        target_group = set(target_leaf.group_ids)
        args = []
        for i, l in enumerate(leaves):
            if i != target_idx and set(l.group_ids) & target_group:
                args.append(f"[{i}]")
        col_name = target_leaf.path[-1] if target_leaf.path else "?"
        return col_name, args

    # 显示 mask 预测 vs 真实值
    print(f"\n  🎯 Masked predictions ({len(masks)} masks):")
    sample_out = out[sample_idx]

    for idx, (truth_val, truth_type) in masks.items():
        mask_repr = sample_out[idx].unsqueeze(0)
        orig_leaf = leaves[idx]
        col_name, args = _find_context(orig_leaf, idx)

        prefix = f"{col_name}({','.join(args[:5])})=[{idx}]"

        if truth_type == "number":
            pred_raw = model.decode_head.predict_number(mask_repr).item()
            truth_f = float(truth_val)
            err = abs(pred_raw - truth_f)
            rel_err = err / (abs(truth_f) + 1e-8)
            print(f"    {prefix} number: pred={pred_raw:12.4f}  true={truth_f:12.4f}  "
                  f"err={err:.4f} ({rel_err:.1%})")
        elif truth_type == "boolean":
            pred_logit = model.decode_head.predict_boolean(mask_repr).item()
            pred_bool = pred_logit > 0
            print(f"    {prefix}   bool: pred={pred_bool} (logit={pred_logit:.3f})  "
                  f"true={truth_val}")
        elif truth_type == "string":
            pred_emb = model.decode_head.predict_string(mask_repr)
            target_emb = model.frozen_lm.encode([truth_val])
            cos_sim = torch.nn.functional.cosine_similarity(pred_emb, target_emb).item()
            print(f"    {prefix} string: cos_sim={cos_sim:.4f}  true=\"{truth_val}\"")

    print(f"  {'─'*60}\n")





def train_stage(
    model: DocumentTransformer,
    frozen_lm: FrozenLM,
    stage_name: str,
    dataset,
    max_epochs: int,
    patience: int,
    train_config: TrainConfig,
    model_config: ModelConfig,
    checkpoint_dir: str = "checkpoints",
    writer=None,
    global_step: int = 0,
    checkpoint_interval: int = 5,
    resume_ckpt: dict = None,
    test_dataset=None,
    eval_metric: str = "rmse",
) -> dict:
    """
    训练单个 Stage，基于 loss plateau 自动结束。

    Args:
        dataset: 数据集实例 (TabularDataset 或 MatbenchDataset)
        max_epochs: 该 Stage 最大 epoch 数上限
        patience: 连续 N 个 epoch loss 不下降则视为收敛
        resume_ckpt: 恢复用的 checkpoint dict

    Returns:
        stage_log: 包含 loss 历史的 dict
    """
    device = train_config.device
    dataset_size = len(dataset)
    print(f"\n{'='*70}")
    print(f"  {stage_name}")
    print(f"  n_rows={getattr(dataset, 'n_rows', 1)}, dataset_size={dataset_size}")
    print(f"  max_epochs={max_epochs}, patience={patience}")
    print(f"{'='*70}\n")

    optimizer = AdamW(model.parameters(), lr=train_config.lr, 
                      weight_decay=train_config.weight_decay, betas=train_config.betas)
    
    # 预估总步数和 Warmup 步数
    steps_per_epoch = dataset_size // train_config.batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = min(625, total_steps // 10)  # 固定 1 epoch warmup，不随 max_epochs 膨胀
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # BF16 混精度训练 — 提速 ~2x，动态范围与 FP32 相同
    # BF16 的指数位与 FP32 相同 → 无下溢问题 → 不需要 GradScaler
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    if use_amp:
        print(f"  ⚡ BF16 mixed precision enabled (GradScaler disabled — not needed for BF16)")

    # 恢复训练状态
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])
        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"]
        if "best_loss" in resume_ckpt:
            best_loss = resume_ckpt["best_loss"]
        if "global_step" in resume_ckpt:
            global_step = resume_ckpt["global_step"]
        
        # 恢复 scheduler 状态：优先使用 state_dict（精确），fallback 到循环快进
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            resume_steps = start_epoch * steps_per_epoch
            for _ in range(resume_steps):
                scheduler.step()
        # 恢复随机状态，确保数据生成和 dropout 等完全一致
        if "rng_states" in resume_ckpt:
            set_rng_states(resume_ckpt["rng_states"])
        resumed_lr = optimizer.param_groups[0]["lr"]
        print(f"  🔄 Resumed: epoch={start_epoch}, best_loss={best_loss:.6f}, "
              f"global_step={global_step}, lr={resumed_lr:.2e}")

    stage_log = {
        "stage": stage_name,
        "losses": [],
        "epoch_times": [],
        "best_loss": best_loss,
        "stopped_at_epoch": 0,
        "reason": "",
    }

    # num_workers=0：预计算 fork_bias 后 collate 极轻量（pad+stack），
    # 多进程 fork 反而因 COW 复制预解析数据集（数百万 Python 对象）而严重拖慢启动
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_fn, max_tokens=model_config.max_tokens),
        num_workers=train_config.num_workers,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=functools.partial(collate_fn, max_tokens=model_config.max_tokens),
            num_workers=0,
        )

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (batched_leaves, batched_masks, padding_mask, fork_bias_indices) in enumerate(loader):
            padding_mask = padding_mask.to(device)
            fork_bias_indices = fork_bias_indices.to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batched_leaves, padding_mask, fork_bias_indices=fork_bias_indices)
                loss = model.compute_loss(out, batched_leaves, batched_masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_batches += 1
            global_step += 1

            # ── TensorBoard: per-step metrics ──
            if writer is not None:
                writer.add_scalar(f"{stage_name}/batch_loss", batch_loss, global_step)
                writer.add_scalar(f"{stage_name}/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar(f"{stage_name}/grad_norm", grad_norm.item(), global_step)

            # ── 每 batch 输出 loss ──
            avg_tokens = sum(len(l) for l in batched_leaves) / len(batched_leaves)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{time.strftime('%H:%M:%S')}] [{stage_name}] E{epoch+1:03d} "
                  f"B{batch_idx:04d}/{len(loader)} "
                  f"Loss={batch_loss:.6f} "
                  f"tokens={avg_tokens:.0f} "
                  f"lr={lr_now:.2e}")
                      
            scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(n_batches, 1)
        stage_log["losses"].append(avg_loss)
        stage_log["epoch_times"].append(epoch_time)

        # GPU 监控
        if device == "cuda":
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            mem_info = f"  GPU: {mem_alloc:.1f}G/peak {mem_peak:.1f}G"
        else:
            mem_info = ""

        # ── Patience 检查 ──
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
            improvement = "↓ new best"
        else:
            patience_counter += 1
            improvement = f"↔ no improve ({patience_counter}/{patience})"

        print(f"  [{stage_name}] E{epoch+1:03d} "
              f"AvgLoss={avg_loss:.6f} ({improvement}) "
              f"Time={epoch_time:.1f}s"
              f"{mem_info}")

        # ── 每 epoch 结束展示一个诊断案例 ──
        _log_sample_case(model, out, batched_leaves, batched_masks,
                         sample_idx=0, stage_name=stage_name,
                         epoch=epoch+1, batch_idx=batch_idx)

        # ── Test 评估 ──
        if test_loader is not None:
            test_score = evaluate(model, test_loader, device, metric=eval_metric)
            metric_name = f"test_{eval_metric}"
            stage_log.setdefault(metric_name, []).append(test_score)
            print(f"  🎯 Test {eval_metric.upper()}: {test_score:.4f}")
            if writer is not None:
                writer.add_scalar(f"{stage_name}/{metric_name}", test_score, epoch + 1)

        # ── TensorBoard: per-epoch metrics ──
        if writer is not None:
            writer.add_scalar(f"{stage_name}/epoch_avg_loss", avg_loss, epoch + 1)
            writer.add_scalar(f"{stage_name}/epoch_time", epoch_time, epoch + 1)
            writer.add_scalar(f"{stage_name}/best_loss", best_loss, epoch + 1)
            if device == "cuda":
                writer.add_scalar(f"{stage_name}/gpu_mem_GiB", mem_alloc, epoch + 1)
                writer.add_scalar(f"{stage_name}/gpu_peak_GiB", mem_peak, epoch + 1)
            writer.flush()

        # ── 定期保存 checkpoint（含完整训练状态）──
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, f"{stage_name}_e{epoch+1}",
                            checkpoint_dir,
                            optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, epoch=epoch+1,
                            best_loss=best_loss, stage_name=stage_name,
                            global_step=global_step)

        # ── 收敛退出 ──
        if patience_counter >= patience:
            print(f"\n  ⏹️  {stage_name} converged! "
                  f"No improvement for {patience} epochs. "
                  f"Best loss: {best_loss:.6f}")
            stage_log["reason"] = f"converged (patience={patience})"
            break
    else:
        print(f"\n  ⏹️  {stage_name} reached max_epochs ({max_epochs}).")
        stage_log["reason"] = f"max_epochs ({max_epochs})"

    stage_log["best_loss"] = best_loss
    stage_log["stopped_at_epoch"] = epoch + 1

    return stage_log, global_step


def save_checkpoint(model, name, checkpoint_dir, log=None,
                    optimizer=None, scheduler=None, scaler=None,
                    epoch=None, best_loss=None, stage_name=None,
                    global_step=None):
    """保存模型 checkpoint，含训练状态以支持 resume。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{name}.pth")
    ckpt = {"model": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if best_loss is not None:
        ckpt["best_loss"] = best_loss
    if stage_name is not None:
        ckpt["stage"] = stage_name
    if global_step is not None:
        ckpt["global_step"] = global_step
    # 保存随机状态，确保 resume 后完全可复现
    ckpt["rng_states"] = get_rng_states()
    torch.save(ckpt, ckpt_path)
    print(f"  💾 Checkpoint: {ckpt_path}")
    if log is not None:
        log_path = os.path.join(checkpoint_dir, f"{name}_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  📄 Log: {log_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Document Transformer",
                                     allow_abbrev=False)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    parser.add_argument("--dataset", type=str, default="california_housing",
                        help="Dataset name (california_housing, matbench_dielectric, etc.)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to custom CSV file (overrides --dataset)")
    # Matbench 特有选项（由 data.matbench.configs 模块管理）
    from data.matbench.configs import register_args as register_matbench_args
    register_matbench_args(parser)
    parser.add_argument("--warm-restart", action="store_true", default=False,
                        help="With --resume: only load model weights, discard optimizer/scheduler/RNG")
    args = parser.parse_args()

    set_seed(SEED)

    model_config = ModelConfig()
    train_config = TrainConfig()

    is_matbench = args.dataset.startswith("matbench_")

    if is_matbench:
        # ═══════════════════════════════════════════
        # Matbench 路径
        # ═══════════════════════════════════════════
        from data.matbench import MatbenchLoader, MatbenchDataset
        from data.matbench.configs import get_matbench_config, build_dataset_options, build_cache_tag

        mb_config = get_matbench_config(args.dataset)
        eval_metric = mb_config.metric  # "mae"

        # 合并 CLI 选项到 dataset_options（逻辑由 configs 模块管理）
        dataset_options = build_dataset_options(args, mb_config)

        print(f"Model: d={model_config.d_model}, L={model_config.n_layers}, "
              f"H={model_config.n_heads}, ff={model_config.d_ff}")
        print(f"Train: bs={train_config.batch_size}, lr={train_config.lr}")
        print(f"Matbench task: {args.dataset} — {mb_config.description}")
        print(f"Eval metric: {eval_metric.upper()}")
        if dataset_options:
            print(f"Dataset options: {dataset_options}")

        mb_loader = MatbenchLoader(args.dataset,
                                   max_tokens=model_config.max_tokens,
                                   dataset_options=dataset_options)
        stages = mb_config.stages

    else:
        # ═══════════════════════════════════════════
        # Tabular 路径 (保持不变)
        # ═══════════════════════════════════════════
        from data.tabular.configs import get_dataset_config
        ds_config = get_dataset_config(args.dataset)
        eval_metric = "rmse"

        print(f"Model: d={model_config.d_model}, L={model_config.n_layers}, "
              f"H={model_config.n_heads}, ff={model_config.d_ff}")
        print(f"Train: bs={train_config.batch_size}, lr={train_config.lr}")
        print(f"Dataset config: n_rows={ds_config.n_rows}, "
              f"stages={len(ds_config.stages)}, task={ds_config.task_type}")

        if args.csv:
            print(f"\n  📂 Loading CSV: {args.csv}")
            full_loader = TableLoader.from_csv(args.csv)
        else:
            print(f"\n  📂 Loading dataset: {args.dataset}")
            full_loader = TableLoader.from_builtin(
                args.dataset, max_rows=ds_config.max_rows,
                target_col=ds_config.target_col,
            )

        print(f"  ✅ {full_loader.n_rows} rows × {full_loader.n_cols} cols")

        train_loader, test_loader = full_loader.split(test_ratio=0.2, seed=42)
        table_loader = train_loader

        print(f"  Numeric: {table_loader.numeric_cols}")
        print(f"  Target: {table_loader.target_col}")
        stages = ds_config.stages

    device = train_config.device
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)

    # Resume
    resume_ckpt = None
    if args.resume:
        print(f"\n  \U0001f504 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if args.warm_restart:
            # Warm restart: 只加载模型权重，optimizer/scheduler/RNG 全部重新初始化
            resume_ckpt = None
            print(f"  🔥 Warm restart: loaded weights only (fresh optimizer/scheduler)")
        else:
            resume_ckpt = ckpt
        print(f"  \u2705 Loaded. epoch={ckpt.get('epoch', '?')}, "
              f"best_loss={ckpt.get('best_loss', '?')}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {total_params:,} ({total_params/1e6:.1f}M)")

    # ── 构建 run_name：{dataset}{opts_tag}_{timestamp} ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    opts_tag = ""
    if is_matbench:
        from data.matbench.configs import build_cache_tag
        opts_tag = build_cache_tag(dataset_options)
    run_name = f"{args.dataset}{opts_tag}_{timestamp}"
    if args.resume:
        run_name += "_resumed"
    checkpoint_dir = os.path.join(train_config.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"  📊 TensorBoard: runs/{run_name}")
    print(f"     Launch: tensorboard --logdir runs/")

    writer.add_text("config/model", f"d={model_config.d_model}, L={model_config.n_layers}, "
                     f"H={model_config.n_heads}, ff={model_config.d_ff}")
    writer.add_text("config/train", f"bs={train_config.batch_size}, lr={train_config.lr}, "
                     f"dataset={args.dataset}")
    if is_matbench:
        opts_str = ", ".join(f"{k}={v}" for k, v in sorted(dataset_options.items())) or "default"
        writer.add_text("config/dataset_options", opts_str)

    all_logs = {}

    # ── 按配方中的 stages 逐阶段训练 ──
    global_step = 0
    for stage_idx, stage_cfg in enumerate(stages):
        if is_matbench:
            # 构建含选项的 cache_tag（逻辑由 configs 模块管理）
            opts_tag = build_cache_tag(dataset_options)

            # Matbench: 每个 doc 是独立嵌套 JSON
            dataset = MatbenchDataset(
                docs=mb_loader.train_docs,
                max_tokens=model_config.max_tokens,
                cache_tag=f"{args.dataset}{opts_tag}_train",
            )
            test_ds = MatbenchDataset(
                docs=mb_loader.test_docs,
                max_tokens=model_config.max_tokens,
                cache_tag=f"{args.dataset}{opts_tag}_test",
            )
        else:
            # Tabular: 多行表格 + BallTree 邻居
            dataset = TabularDataset(
                loader=table_loader,
                n_rows=ds_config.n_rows,
                mask_ratio=train_config.mask_ratio,
                max_tokens=model_config.max_tokens,
            )
            test_ds = TabularDataset(
                loader=test_loader,
                n_rows=ds_config.n_rows,
                mask_ratio=train_config.mask_ratio,
                max_tokens=model_config.max_tokens,
            )

        stage_name = f"{args.dataset}_{stage_cfg.name}"
        log, global_step = train_stage(
            model=model,
            frozen_lm=frozen_lm,
            stage_name=stage_name,
            dataset=dataset,
            max_epochs=stage_cfg.max_epochs,
            patience=stage_cfg.patience,
            train_config=train_config,
            model_config=model_config,
            checkpoint_dir=checkpoint_dir,
            writer=writer,
            global_step=global_step,
            resume_ckpt=resume_ckpt if stage_idx == 0 else None,
            test_dataset=test_ds,
            eval_metric=eval_metric,
        )
        save_checkpoint(model, f"{stage_cfg.name}_final", checkpoint_dir, log)
        all_logs[stage_cfg.name] = log

    # ── 保存日志 ──
    log_path = os.path.join(checkpoint_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\n  📁 Log: {log_path}")

    plot_loss_curves(all_logs, checkpoint_dir)

    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"{'='*70}")
    for name, log in all_logs.items():
        print(f"  {log['stage']:25s} | "
              f"epochs={log['stopped_at_epoch']:3d} | "
              f"best_loss={log['best_loss']:.6f} | {log['reason']}")
    print(f"{'='*70}\n")

    writer.close()


if __name__ == "__main__":
    main()

