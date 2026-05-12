"""
课程学习训练入口 — 按 plan.md 第八章执行三阶段训练。

  Stage 1: 显式规则基础训练 (explicit)
  Stage 2: In-Context 上下文规则归纳 (in_context)
  Stage 3: 混合鲁棒性训练 (mixed)

特性：
  - 基于 loss 收敛自动切换阶段 (patience-based)
  - 每个 Stage 在 loss 不再下降时自动结束
  - Loss 曲线自动保存为 PNG 图片
  - 定期保存 checkpoint (每 N epochs)
"""

import os
import time
import json
import torch
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from data.synthetic import SyntheticDataset, collate_fn


def plot_loss_curves(all_logs: dict, save_dir: str):
    """绘制所有 Stage 的 loss 曲线并保存为 PNG。"""
    os.makedirs(save_dir, exist_ok=True)

    # ── 分阶段子图 ──
    fig, axes = plt.subplots(1, len(all_logs), figsize=(6 * len(all_logs), 5))
    if len(all_logs) == 1:
        axes = [axes]

    colors = {"stage1": "#4CAF50", "stage2": "#2196F3", "stage3": "#FF9800"}

    for ax, (stage_key, log) in zip(axes, all_logs.items()):
        losses = log["losses"]
        color = colors.get(stage_key, "#666")
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

    for stage_key, log in all_logs.items():
        losses = log["losses"]
        color = colors.get(stage_key, "#666")
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


def train_stage(
    model: DocumentTransformer,
    frozen_lm: FrozenLM,
    stage_name: str,
    train_mode: str,
    max_epochs: int,
    patience: int,
    dataset_size: int,
    target_tokens: int,
    distractor_level: int,
    train_config: TrainConfig,
    model_config: ModelConfig,
    checkpoint_interval: int = 5,
) -> dict:
    """
    训练单个 Stage，基于 loss plateau 自动结束。

    Args:
        max_epochs: 该 Stage 最大 epoch 数上限（保底退出）
        patience: 连续 N 个 epoch loss 不下降则视为收敛
        checkpoint_interval: 每 N 个 epoch 保存一次 checkpoint

    Returns:
        stage_log: 包含 loss 历史的 dict
    """
    device = train_config.device
    print(f"\n{'='*70}")
    print(f"  {stage_name}")
    print(f"  mode={train_mode}, max_epochs={max_epochs}, patience={patience}")
    print(f"  dataset_size={dataset_size}, target_tokens={target_tokens}")
    print(f"  distractor_level={distractor_level}")
    print(f"{'='*70}\n")

    optimizer = AdamW(model.parameters(), lr=train_config.lr, 
                      weight_decay=0.01, betas=(0.9, 0.95))
    
    # 预估总步数和 Warmup 步数
    steps_per_epoch = dataset_size // train_config.batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.05) # 5% 的 warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # BF16 混精度训练 — 提速 ~2x，动态范围与 FP32 相同
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print(f"  ⚡ BF16 mixed precision enabled")

    stage_log = {
        "stage": stage_name,
        "losses": [],
        "epoch_times": [],
        "best_loss": float("inf"),
        "stopped_at_epoch": 0,
        "reason": "",
    }
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # 每个 epoch 生成新数据
        dataset = SyntheticDataset(
            size=dataset_size,
            mask_ratio=train_config.mask_ratio,
            max_tokens=model_config.max_tokens,
            train_mode=train_mode,
            target_tokens=target_tokens,
            distractor_level=distractor_level,
        )
        loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, model_config.max_tokens),
            num_workers=0,
        )

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (batched_leaves, batched_masks, padding_mask) in enumerate(loader):
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batched_leaves, padding_mask)
                loss = model.compute_loss(out, batched_leaves, batched_masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_batches += 1

            if batch_idx % 50 == 0:
                avg_tokens = sum(len(l) for l in batched_leaves) / len(batched_leaves)
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  [{stage_name}] E{epoch+1:03d} "
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

        # ── 定期保存 checkpoint ──
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, f"{stage_name}_e{epoch+1}",
                            train_config.checkpoint_dir)

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

    return stage_log


def save_checkpoint(model, name, checkpoint_dir, log=None):
    """保存模型 checkpoint。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  💾 Checkpoint: {ckpt_path}")

    if log is not None:
        log_path = os.path.join(checkpoint_dir, f"{name}_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)


def main():
    model_config = ModelConfig()
    train_config = TrainConfig()

    print(f"Model: d={model_config.d_model}, L={model_config.n_layers}, "
          f"H={model_config.n_heads}, ff={model_config.d_ff}")
    print(f"Train: bs={train_config.batch_size}, lr={train_config.lr}")

    device = train_config.device
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Checkpoint dir: {train_config.checkpoint_dir}")

    all_logs = {}

    # ════════════════════════════════════════
    # Stage 1: 显式规则基础训练
    # ════════════════════════════════════════
    log1 = train_stage(
        model=model,
        frozen_lm=frozen_lm,
        stage_name="stage1_explicit",
        train_mode="explicit",
        max_epochs=train_config.stage1_max_epochs,
        patience=train_config.stage1_patience,
        dataset_size=train_config.dataset_size,
        target_tokens=train_config.stage1_target_tokens,
        distractor_level=train_config.stage1_distractor_level,
        train_config=train_config,
        model_config=model_config,
    )
    save_checkpoint(model, "stage1_final", train_config.checkpoint_dir, log1)
    all_logs["stage1"] = log1

    # ════════════════════════════════════════
    # Stage 2: In-Context 上下文规则归纳
    # ════════════════════════════════════════
    log2 = train_stage(
        model=model,
        frozen_lm=frozen_lm,
        stage_name="stage2_in_context",
        train_mode="in_context",
        max_epochs=train_config.stage2_max_epochs,
        patience=train_config.stage2_patience,
        dataset_size=train_config.dataset_size,
        target_tokens=train_config.stage2_target_tokens,
        distractor_level=train_config.stage2_distractor_level,
        train_config=train_config,
        model_config=model_config,
    )
    save_checkpoint(model, "stage2_final", train_config.checkpoint_dir, log2)
    all_logs["stage2"] = log2

    # ════════════════════════════════════════
    # Stage 3: 混合鲁棒性训练
    # ════════════════════════════════════════
    log3 = train_stage(
        model=model,
        frozen_lm=frozen_lm,
        stage_name="stage3_mixed",
        train_mode="mixed",
        max_epochs=train_config.stage3_max_epochs,
        patience=train_config.stage3_patience,
        dataset_size=train_config.dataset_size,
        target_tokens=train_config.stage3_target_tokens,
        distractor_level=train_config.stage3_distractor_level,
        train_config=train_config,
        model_config=model_config,
    )
    save_checkpoint(model, "stage3_final", train_config.checkpoint_dir, log3)
    all_logs["stage3"] = log3

    # ════════════════════════════════════════
    # 保存完整日志 + 绘制 Loss 曲线
    # ════════════════════════════════════════
    log_path = os.path.join(train_config.checkpoint_dir, "full_training_log.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\n  📁 Full log: {log_path}")

    plot_loss_curves(all_logs, train_config.checkpoint_dir)

    # 训练摘要
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"{'='*70}")
    for stage_key, log in all_logs.items():
        print(f"  {log['stage']:25s} | "
              f"epochs={log['stopped_at_epoch']:3d} | "
              f"best_loss={log['best_loss']:.6f} | "
              f"{log['reason']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
