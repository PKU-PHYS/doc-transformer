"""
课程学习训练入口 — 按 plan.md 第八章执行四阶段课程学习训练。

  Stage 0: 极简预热训练 (simple) — Tier 0 扁平 KV
  Stage 1: 复合与树状结构基础训练 (explicit) — 嵌套/复合/数组/文本任务
  Stage 2: 抗噪训练 (explicit_long) — 长文档 + 重度干扰
  Stage 3: 上下文规则归纳 (in_context) — 隐式函数推断

特性：
  - 基于 loss 收敛自动切换阶段 (patience-based)
  - 每个 Stage 在 loss 不再下降时自动结束
  - Loss 曲线自动保存为 PNG 图片
  - 定期保存 checkpoint (每 N epochs)
  - 支持 --resume 断点续训，含完整 RNG 状态恢复
"""

import os
import time
import json
import random
import argparse
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
from data.synthetic import SyntheticDataset, collate_fn


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
        torch.random.set_rng_state(states["torch_rng"])
    if "cuda_rng" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda_rng"])


def plot_loss_curves(all_logs: dict, save_dir: str):
    """绘制所有 Stage 的 loss 曲线并保存为 PNG。"""
    os.makedirs(save_dir, exist_ok=True)

    # ── 分阶段子图 ──
    fig, axes = plt.subplots(1, len(all_logs), figsize=(6 * len(all_logs), 5))
    if len(all_logs) == 1:
        axes = [axes]

    colors = {"stage0": "#9C27B0", "stage1": "#4CAF50", "stage2": "#2196F3", "stage3": "#FF9800"}

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
    
    from data.functions.registry import FUNC_NAME_KEYS
    
    # 将 leaves 分为 useful (与 mask 属于同一 group) 和 useless (distractors)
    masked_group_ids = set()
    for idx in masks.keys():
        masked_group_ids.update(leaves[idx].group_ids)
        
    useful_indices = []
    useless_indices = []
    for i, l in enumerate(leaves):
        if set(l.group_ids) & masked_group_ids or not masked_group_ids:
            # 如果没有 group (极简情况) 或者匹配到了 group
            useful_indices.append(i)
        else:
            useless_indices.append(i)

    def _format_leaf(i, leaf, is_masked):
        path_str = ".".join(leaf.path[-2:]) if len(leaf.path) > 2 else ".".join(leaf.path)
        marker = "  🎯" if is_masked else ""
        if leaf.value_type == "number":
            val_str = f"{leaf.value:.4g}"
        elif leaf.value_type == "string":
            val_str = f'"{leaf.value[:20]}"' if len(str(leaf.value)) > 20 else f'"{leaf.value}"'
        else:
            val_str = str(leaf.value)
        return f"    [{i:3d}] {path_str:30s} = {val_str:15s} ({leaf.value_type}){marker}"

    print(f"  🟢 Useful Leaves (associated with masks) ({len(useful_indices)} tokens):")
    for i in useful_indices:
        print(_format_leaf(i, leaves[i], i in masks))
        
    if useless_indices:
        print(f"  🔴 Distractor Leaves ({len(useless_indices)} tokens):")
        for i in useless_indices[:5]:
            print(_format_leaf(i, leaves[i], False))
        if len(useless_indices) > 5:
            print(f"    ... ({len(useless_indices) - 5} more distractor tokens)")
            
    # 智能推断函数上下文的辅助函数
    def _find_func_context_and_args(target_leaf, target_idx):
        target_group = set(target_leaf.group_ids)
        
        # 收集在同一个大 group 里的所有节点（同组依赖）
        siblings = [l for l in leaves if set(l.group_ids) & target_group]
        # 如果是 Stage 0，没生成有效 group，或者没找到
        if not siblings:
            siblings = leaves
            
        func_name = "implicit_function"
        args = []
        
        for l in siblings:
            idx = leaves.index(l)
            path_key = l.path[-1].lower()
            if l.value_type == "string" and path_key in FUNC_NAME_KEYS:
                func_name = str(l.value)
            elif idx != target_idx and path_key not in ["category", "id", "timestamp", "author", "confidence", "tags", "label", "family", "group"]:
                args.append(f"[{idx}]")
                
        return func_name, args

    # 显示 mask 预测 vs 真实值
    print(f"\n  🎯 Masked predictions ({len(masks)} masks):")
    sample_out = out[sample_idx]  # (max_len, d_model)
    
    for idx, (truth_val, truth_type) in masks.items():
        mask_repr = sample_out[idx].unsqueeze(0)
        orig_leaf = leaves[idx]
        func_name, args = _find_func_context_and_args(orig_leaf, idx)
        
        prefix = f"{func_name}({','.join(args)})=[{idx}]"
        
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
    train_mode: str,
    max_epochs: int,
    patience: int,
    dataset_size: int,
    target_tokens: int,
    distractor_level: int,
    train_config: TrainConfig,
    model_config: ModelConfig,
    writer: SummaryWriter = None,
    global_step: int = 0,
    checkpoint_interval: int = 5,
    resume_ckpt: dict = None,
) -> dict:
    """
    训练单个 Stage，基于 loss plateau 自动结束。

    Args:
        max_epochs: 该 Stage 最大 epoch 数上限（保底退出）
        patience: 连续 N 个 epoch loss 不下降则视为收敛
        checkpoint_interval: 每 N 个 epoch 保存一次 checkpoint
        resume_ckpt: 恢复用的 checkpoint dict，含 optimizer/scheduler/scaler/epoch/best_loss

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
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print(f"  ⚡ BF16 mixed precision enabled")

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

    for epoch in range(start_epoch, max_epochs):
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

            # ── 诊断输出：每 200 batch 展示一个案例 ──
            if batch_idx % 200 == 0 and batch_idx > 0:
                _log_sample_case(model, out, batched_leaves, batched_masks, 
                                 sample_idx=0, stage_name=stage_name, 
                                 epoch=epoch+1, batch_idx=batch_idx)

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
                            train_config.checkpoint_dir,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()

    # 全局 seed — 保证每次从头跑都完全一致
    set_seed(SEED)

    model_config = ModelConfig()
    train_config = TrainConfig()

    print(f"Model: d={model_config.d_model}, L={model_config.n_layers}, "
          f"H={model_config.n_heads}, ff={model_config.d_ff}")
    print(f"Train: bs={train_config.batch_size}, lr={train_config.lr}")

    device = train_config.device
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)

    # Resume: 加载 checkpoint
    resume_stage = None
    resume_ckpt = None
    if args.resume:
        print(f"\n  \U0001f504 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        resume_stage = ckpt.get("stage", None)
        # 统一为短名: "stage0_simple" → "stage0"
        if resume_stage and "stage0" in resume_stage:
            resume_stage = "stage0"
        elif resume_stage and "stage1" in resume_stage:
            resume_stage = "stage1"
        elif resume_stage and "stage2" in resume_stage:
            resume_stage = "stage2"
        elif resume_stage and "stage3" in resume_stage:
            resume_stage = "stage3"
        resume_ckpt = ckpt
        print(f"  \u2705 Loaded. Stage={resume_stage}, epoch={ckpt.get('epoch', '?')}, "
              f"best_loss={ckpt.get('best_loss', '?')}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Checkpoint dir: {train_config.checkpoint_dir}")

    # ── TensorBoard ──
    run_name = time.strftime("%Y%m%d_%H%M%S")
    if args.resume:
        run_name += "_resumed"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"  📊 TensorBoard: runs/{run_name}")
    print(f"     Launch: tensorboard --logdir runs/")

    # 记录超参数
    writer.add_text("config/model", f"d={model_config.d_model}, L={model_config.n_layers}, "
                     f"H={model_config.n_heads}, ff={model_config.d_ff}")
    writer.add_text("config/train", f"bs={train_config.batch_size}, lr={train_config.lr}, "
                     f"beta2=0.95, bf16=True")

    all_logs = {}
    global_step = 0

    # 确定要跳过的 stages（resume 时从下一个 stage 开始）
    skip_stages = set()
    if resume_stage == "stage0":
        pass
    elif resume_stage == "stage1":
        skip_stages.add("stage0")
    elif resume_stage == "stage2":
        skip_stages.update(["stage0", "stage1"])
    elif resume_stage == "stage3":
        skip_stages.update(["stage0", "stage1", "stage2"])

    # ════════════════════════════════════════
    # Stage 0: 极简预热训练
    # ════════════════════════════════════════
    if "stage0" not in skip_stages:
        log0, global_step = train_stage(
            model=model,
            frozen_lm=frozen_lm,
            stage_name="stage0_simple",
            train_mode="simple",
            max_epochs=train_config.stage0_max_epochs,
            patience=train_config.stage0_patience,
            dataset_size=train_config.dataset_size,
            target_tokens=train_config.stage0_target_tokens,
            distractor_level=train_config.stage0_distractor_level,
            train_config=train_config,
            model_config=model_config,
            writer=writer,
            global_step=global_step,
            resume_ckpt=resume_ckpt if resume_stage == "stage0" else None,
        )
        save_checkpoint(model, "stage0_final", train_config.checkpoint_dir, log0)
        all_logs["stage0"] = log0
    else:
        print(f"\n  ⏭️  Skipping stage0 (resumed from {resume_stage})")

    # ════════════════════════════════════════
    # Stage 1: 复合与树状结构基础训练
    # ════════════════════════════════════════
    if "stage1" not in skip_stages:
        log1, global_step = train_stage(
            model=model,
            frozen_lm=frozen_lm,
            stage_name="stage1_composite",
            train_mode="explicit",
            max_epochs=train_config.stage1_max_epochs,
            patience=train_config.stage1_patience,
            dataset_size=train_config.dataset_size,
            target_tokens=train_config.stage1_target_tokens,
            distractor_level=train_config.stage1_distractor_level,
            train_config=train_config,
            model_config=model_config,
            writer=writer,
            global_step=global_step,
            resume_ckpt=resume_ckpt if resume_stage == "stage1" else None,
        )
        save_checkpoint(model, "stage1_final", train_config.checkpoint_dir, log1)
        all_logs["stage1"] = log1
    else:
        print(f"\n  ⏭️  Skipping stage1 (resumed from {resume_stage})")

    # ════════════════════════════════════════
    # Stage 2: 抗噪训练 (长文档与干扰字段)
    # ════════════════════════════════════════
    if "stage2" not in skip_stages:
        log2, global_step = train_stage(
            model=model,
            frozen_lm=frozen_lm,
            stage_name="stage2_antinoise",
            train_mode="explicit_long",
            max_epochs=train_config.stage2_max_epochs,
            patience=train_config.stage2_patience,
            dataset_size=train_config.dataset_size,
            target_tokens=train_config.stage2_target_tokens,
            distractor_level=train_config.stage2_distractor_level,
            train_config=train_config,
            model_config=model_config,
            writer=writer,
            global_step=global_step,
            resume_ckpt=resume_ckpt if resume_stage == "stage2" else None,
        )
        save_checkpoint(model, "stage2_final", train_config.checkpoint_dir, log2)
        all_logs["stage2"] = log2
    else:
        print(f"\n  ⏭️  Skipping stage2 (resumed from {resume_stage})")

    # ════════════════════════════════════════
    # Stage 3: 上下文规则归纳 (In-Context Learning)
    # ════════════════════════════════════════
    if "stage3" not in skip_stages:
        log3, global_step = train_stage(
            model=model,
            frozen_lm=frozen_lm,
            stage_name="stage3_incontext",
            train_mode="in_context",
            max_epochs=train_config.stage3_max_epochs,
            patience=train_config.stage3_patience,
            dataset_size=train_config.dataset_size,
            target_tokens=train_config.stage3_target_tokens,
            distractor_level=train_config.stage3_distractor_level,
            train_config=train_config,
            model_config=model_config,
            writer=writer,
            global_step=global_step,
            resume_ckpt=resume_ckpt if resume_stage == "stage3" else None,
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

    writer.close()


if __name__ == "__main__":
    main()
