from __future__ import annotations

import csv
import os

import torch
import torch.nn.functional as F

import usim_feedback_fast3_content_delta as legacy

# Static runner guard tokens delegated to legacy: def run_static_experiment, _static_split_df


def forced_cold_distribution_consistency_loss(
    teacher_logits,
    student_logits,
    *,
    active_rows=None,
    invalid_candidate_mask=None,
    temperature=0.20,
):
    """Return one-way KL from detached warm targets to forced-cold scores."""
    if not isinstance(teacher_logits, torch.Tensor) or not isinstance(
        student_logits,
        torch.Tensor,
    ):
        raise ValueError("teacher_logits and student_logits must be tensors")
    if teacher_logits.ndim != 2 or student_logits.ndim != 2:
        raise ValueError("teacher_logits and student_logits must be 2-D")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have identical shapes")

    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be greater than zero") from exc
    if not temperature > 0.0:
        raise ValueError("temperature must be greater than zero")

    batch_size = teacher_logits.shape[0]
    if active_rows is None:
        active_rows = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=student_logits.device,
        )
    else:
        if (
            not isinstance(active_rows, torch.Tensor)
            or active_rows.dtype != torch.bool
            or active_rows.shape != (batch_size,)
        ):
            raise ValueError("active_rows must be boolean with shape [batch]")
        active_rows = active_rows.to(device=student_logits.device)

    if invalid_candidate_mask is None:
        invalid_candidate_mask = torch.zeros_like(
            student_logits,
            dtype=torch.bool,
        )
    else:
        if (
            not isinstance(invalid_candidate_mask, torch.Tensor)
            or invalid_candidate_mask.dtype != torch.bool
            or invalid_candidate_mask.shape != teacher_logits.shape
        ):
            raise ValueError(
                "invalid_candidate_mask must be boolean and match the logits shape",
            )
        invalid_candidate_mask = invalid_candidate_mask.to(
            device=student_logits.device,
        )

    if not active_rows.any().item():
        return student_logits.sum() * 0.0

    active_invalid_mask = invalid_candidate_mask[active_rows]
    if active_invalid_mask.all(dim=1).any().item():
        raise ValueError("an active row cannot have every candidate masked")

    teacher_scaled = teacher_logits.detach()[active_rows] / temperature
    student_scaled = student_logits[active_rows] / temperature
    teacher_scaled = teacher_scaled.masked_fill(active_invalid_mask, -torch.inf)
    student_scaled = student_scaled.masked_fill(active_invalid_mask, -torch.inf)

    teacher_targets = F.softmax(teacher_scaled, dim=-1)
    student_log_probs = F.log_softmax(student_scaled, dim=-1)
    candidate_kl = F.kl_div(
        student_log_probs,
        teacher_targets,
        reduction="none",
    ).masked_fill(active_invalid_mask, 0.0)
    return candidate_kl.sum(dim=-1).mean() * (temperature**2)


class SC2ConsistencyConfig(legacy.Fast3Config):
    """Isolated controls for the SC2Rec-style consistency experiment."""

    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.sc2_consistency_weight = float(
            os.environ.get("USIM_SC2_CONSISTENCY_WEIGHT", "0.10"),
        )
        self.sc2_consistency_temp = float(
            os.environ.get("USIM_SC2_CONSISTENCY_TEMP", "0.20"),
        )
        self.sc2_consistency_warm_only = (
            os.environ.get("USIM_SC2_CONSISTENCY_WARM_ONLY", "1") == "1"
        )
        if self.sc2_consistency_weight < 0.0:
            raise ValueError("USIM_SC2_CONSISTENCY_WEIGHT must be non-negative")
        if self.sc2_consistency_temp <= 0.0:
            raise ValueError("USIM_SC2_CONSISTENCY_TEMP must be greater than zero")


class SC2ConsistencyFast3FeedbackUSIM(legacy.Fast3FeedbackUSIM):
    """CKG-RL with one-way warm-to-forced-cold score distillation."""

    _SC2_DIAGNOSTIC_KEYS = (
        "sc2_consistency_loss",
        "sc2_consistency_weighted_loss",
        "sc2_consistency_active_ratio",
        "sc2_teacher_student_cosine",
    )

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self._sc2_epoch_sums = {key: 0.0 for key in self._SC2_DIAGNOSTIC_KEYS}
        self._sc2_epoch_batches = 0
        self._sc2_epoch_index = 0

    def train(self, mode=True):
        was_training = bool(self.training)
        result = super().train(mode)
        if was_training and not mode and self._sc2_epoch_batches > 0:
            self._flush_sc2_epoch_metrics()
        return result

    def _flush_sc2_epoch_metrics(self):
        count = max(1, int(self._sc2_epoch_batches))
        self._sc2_epoch_index += 1
        row = {"epoch": self._sc2_epoch_index, "batches": count}
        for key in self._SC2_DIAGNOSTIC_KEYS:
            row[key] = float(self._sc2_epoch_sums.get(key, 0.0)) / count

        output_dir = os.environ.get("USIM_FB_OUTPUT_DIR", ".")
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "sc2_consistency_epoch_metrics.csv")
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(
            "  [SC2-CONSISTENCY] "
            f"epoch={row['epoch']} batches={count} "
            f"loss={row['sc2_consistency_loss']:.6f} "
            f"weighted={row['sc2_consistency_weighted_loss']:.6f} "
            f"active={row['sc2_consistency_active_ratio']:.2%} "
            f"cos={row['sc2_teacher_student_cosine']:.4f}",
        )
        self._sc2_epoch_sums = {key: 0.0 for key in self._SC2_DIAGNOSTIC_KEYS}
        self._sc2_epoch_batches = 0

    def _zero_sc2_result(self):
        zero = self.user_emb.weight.sum() * 0.0
        return zero, {
            "sc2_consistency_loss": 0.0,
            "sc2_consistency_active_ratio": 0.0,
            "sc2_teacher_student_cosine": 0.0,
        }

    def _sc2_consistency_loss(self, batch, pop, llm_s, user_seen_items):
        if not self.training or self.cfg.sc2_consistency_weight <= 0.0:
            return self._zero_sc2_result()

        u, i = batch["u"], batch["i"]
        if u.numel() == 0:
            return self._zero_sc2_result()

        pop_t = torch.as_tensor(pop, dtype=torch.float32, device=i.device).view(-1)
        if pop_t.shape != i.shape:
            raise ValueError("pop must contain one value per target course")
        if self.cfg.sc2_consistency_warm_only:
            active_rows = pop_t >= float(self.cfg.cold_threshold)
        else:
            active_rows = torch.ones_like(pop_t, dtype=torch.bool)
        if not bool(active_rows.any().item()):
            return self._zero_sc2_result()

        with torch.no_grad():
            learner_vec = F.normalize(self.user_proj(self.user_emb(u)), dim=1)
            teacher_item, _, _ = self.get_item_vector(
                i,
                llm_s,
                force_cold=False,
                disable_id_dropout=True,
            )
            teacher_item = F.normalize(teacher_item, dim=1)
            teacher_logits = torch.matmul(learner_vec, teacher_item.t())

        student_item, _, _ = self.get_item_vector(
            i,
            llm_s,
            force_cold=True,
            disable_id_dropout=True,
        )
        student_item = F.normalize(student_item, dim=1)
        student_logits = torch.matmul(learner_vec, student_item.t())

        invalid_candidate_mask = None
        if student_logits.size(0) > 1:
            pos_mask = torch.eye(
                student_logits.size(0),
                dtype=torch.bool,
                device=student_logits.device,
            )
            invalid_candidate_mask = self._build_batch_false_negative_mask(
                u,
                i,
                user_seen_items,
                pos_mask,
            )

        consistency_loss = forced_cold_distribution_consistency_loss(
            teacher_logits,
            student_logits,
            active_rows=active_rows,
            invalid_candidate_mask=invalid_candidate_mask,
            temperature=self.cfg.sc2_consistency_temp,
        )
        paired_cosine = F.cosine_similarity(
            teacher_item.detach(),
            student_item.detach(),
            dim=1,
        )
        diagnostics = {
            "sc2_consistency_loss": float(consistency_loss.detach().item()),
            "sc2_consistency_active_ratio": float(active_rows.float().mean().item()),
            "sc2_teacher_student_cosine": float(paired_cosine[active_rows].mean().item()),
        }
        return consistency_loss, diagnostics

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        base_loss, stats = super().forward(
            batch,
            pop,
            llm_s,
            user_bank_raw=user_bank_raw,
            user_seen_items=user_seen_items,
        )
        consistency_loss, diagnostics = self._sc2_consistency_loss(
            batch,
            pop,
            llm_s,
            user_seen_items,
        )
        weighted_loss = self.cfg.sc2_consistency_weight * consistency_loss
        stats.update(diagnostics)
        stats["sc2_consistency_weighted_loss"] = float(weighted_loss.detach().item())
        if self.training:
            for key in self._SC2_DIAGNOSTIC_KEYS:
                self._sc2_epoch_sums[key] += float(stats.get(key, 0.0))
            self._sc2_epoch_batches += 1
        return base_loss + weighted_loss, stats


def install_sc2_bindings():
    """Install only the experiment config/model into the imported pipeline."""
    legacy.Fast3Config = SC2ConsistencyConfig
    legacy.Fast3FeedbackUSIM = SC2ConsistencyFast3FeedbackUSIM


def main():
    install_sc2_bindings()
    print(">> SC2Rec-style forced-cold consistency entrypoint active")
    print(">> Base pipeline: current CKG-RL strict course-cold implementation")
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
