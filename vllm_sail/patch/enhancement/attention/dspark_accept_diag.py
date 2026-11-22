# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: one-shot diagnostics for DSpark draft acceptance on PPU.

Not a fix and not a behavior change: both hooks run the upstream code first and
only read its inputs/outputs afterwards. Each side reports once, at WARNING
level (the worker drops INFO), then disables itself. Off by default: set
``VLLM_SAIL_DSPARK_DIAG=1`` to install the hooks, and delete this module plus its
conditional import in ``__init__.py`` to remove them.

Both hooks print only when the current stream is not being captured, so a run
that replays CUDA graphs prints nothing: run the diagnostic pass with
``--enforce-eager`` (or with cudagraph mode off) to get numbers. Each report
averages ``_STEPS`` serving steps and ignores the engine's memory-profiling
pass, whose synthetic batch leaves ``sample_pos`` zeroed and whose sequence
lengths are tiny; the draft and verify reports also print their first step's
draft tokens, so the handoff (sampled drafts == verified inputs) can be checked
by eye.

Draft side (``DSparkSpeculator._sample_markov_chain``) answers "is the draft's
own block correct?":

* ``sample_pos`` deltas over the block: the backbone must see positions
  ``P, P+1, ... `` — a delta that is not ``+1`` shifts every later position;
* the cosine similarity of the backbone hidden state between adjacent draft
  positions: rows that are near-identical mean the block forward read the same
  position (or the same KV) for every row;
* the Markov bias against the base logits at each step: its RMS next to the
  logits' RMS, how often it flips the argmax, and how often the corrected
  argmax equals the sampled draft token. A bias that barely moves the argmax
  (candidate "draft ignores its Markov head") shows up here.

* the corrected argmax against the sampled draft token, and the corrected logits
  against the logits the gumbel kernel cached for the same row: the kernel
  caches what it sampled (``logits + bias``), so a gap there means the draft's
  own chain ran on tokens other than the ones it just sampled;
* the block's input ids, positions and query slot mapping: rows after the
  anchor are the mask token at ``P+1, P+2, ...`` and each row attends to the
  rows above it, so a zeroed or shifted position, or a PAD slot, leaves every
  mask row with the context only — the shape a per-position curve shows when it
  decays once and then flattens;
* how far the Markov bias moves when the previous token is replaced, and how
  often that swap changes the corrected argmax: a bias that ignores its token
  leaves the chain with no intra-block signal at all.

Verify side (``RejectionSampler._verify``) answers "is the verify step
faithful?":

* ``positions`` deltas, ``local_pos`` and the accepted tokens per block, to
  confirm the hook sees a serving batch and the layout it assumes;
* the per-position top-1 probability of the target distribution: a target that
  stays confident while acceptance collapses blames the draft tokens, whereas a
  top-1 probability that decays over the block blames the target's multi-query
  attention (the sparse indexer's per-row top-k);
* ``sum min(p, q)`` per position, with ``p`` the target distribution the
  sampler used and ``q`` the cached draft distribution the ratio test divides
  by, is the per-position acceptance this block *should* reach. Block
  verification judges each row on its own, so this is directly comparable to
  the engine's "Per-position acceptance rate" line — same statistic, no
  prefix conditioning. A theoretical value well above the engine's separates an
  under-accepting handoff (the sampler rejecting its own inputs) from a draft
  distribution that is simply worse than the reference;
* ``argmax(target) == draft token`` per position, i.e. how often the draft
  picked the target's own best token: a draft that keeps its top-1 probability
  while this collapses is confidently wrong, not uncertain;
* the draft's top-1 probability and its mass at the target's argmax: a draft
  that stays confident while the overlap collapses sampled a different token,
  not an uncertain one;
* ``argmax(target) == argmax(draft)`` per position: with a greedy draft the two
  argmaxes agree exactly when the draw is accepted, so this is the curve a
  temperature-0 run of the same draft would report and it separates a sampling
  regime difference from a worse draft distribution;
* the same overlap for the backbone alone (cached row minus the Markov bias of
  the token the chain used) and for a teacher-forced chain (the Markov bias of
  the token the target itself prefers): a backbone that stays good while both
  chained values collapse blames the chain's own samples, not the block
  representation. Reported for request 0 only.

``VLLM_SAIL_DSPARK_BIAS_SCALE`` multiplies the Markov bias the chain adds to
every row, which turns a run into a bias-strength sweep without touching the
speculator: ``0`` drops the intra-block signal entirely (the chain falls back
to the plain backbone) and ``1`` is the shipped behaviour. The verify report
prints the active scale, and both reconstructions above stay exact at any
scale because they read the same wrapped method the sampler does.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.34.0,<0.35.0"

_DSPARK_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
_REJECTION_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"

_STEPS = 200
"""Real decode steps averaged into one report.

The engine's first step is the memory-profiling pass, which runs a synthetic
17-token batch with zeroed buffers, so a report must wait for a step that looks
like serving. The window is wide on purpose: a serving batch turns over within
seconds, and the engine's own metrics are steady-state, so only a few hundred
steps put the two on the same footing."""

#: A hook whose key is present has already reported and stops checking.
_reported: set[str] = set()

#: Consecutive exceptions per hook. A hook that keeps raising gives up so a
#: broken measurement cannot flood the log for a whole run.
_failures: dict[str, int] = {}
_MAX_FAILURES = 5

_draft_acc: dict = {}
_verify_acc: dict = {}
# Draft model published by the sampling hook: the rejection sampler keeps no
# handle on it, and the verify hook needs the Markov head to rebuild base
# logits from the cached ones.
_draft_model: list = [None]

_BIAS_SCALE_ENV = "VLLM_SAIL_DSPARK_BIAS_SCALE"

_BIAS_SCALE: float = 1.0
"""Active multiplier for the Markov bias, set from ``_BIAS_SCALE_ENV``.

``1`` is the shipped behaviour and leaves the model untouched. ``0`` drops the
intra-block signal entirely, so the chain becomes the plain backbone; values in
between interpolate. Only the bias the sampler consumes is scaled, which keeps
the engine's acceptance length comparable across runs.
"""


def _resolve_bias_scale() -> float:
    raw = os.getenv(_BIAS_SCALE_ENV, "").strip()
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    # NaN fails both comparisons, so one check covers non-numeric and out of range.
    if not 0.0 <= value <= 8.0:
        logger.warning(
            "Ignoring %s=%r: expected a number in [0, 8].",
            _BIAS_SCALE_ENV,
            raw,
        )
        return 1.0
    return value


def _install_bias_scale(model, scale: float) -> None:
    """Scale the Markov bias the chain adds, once per draft model.

    ``DSparkSpeculator._sample_sequential`` reads the bias from a single call,
    ``self.model.markov_bias``, and either adds it to the logits itself or hands
    it to the fused Gumbel kernel, so wrapping that call covers both paths. The
    verify hook rebuilds base / teacher-forced logits through the same method,
    so it stays consistent with whatever scale is active.
    """
    marker = "_dspark_diag_bias_scale"
    if getattr(model, marker, None) is not None:
        return
    original = model.markov_bias

    def scaled_bias(markov_embed):
        return original(markov_embed) * scale

    model.markov_bias = scaled_bias
    setattr(model, marker, scale)


def _enabled() -> bool:
    return os.getenv("VLLM_SAIL_DSPARK_DIAG", "0").lower() not in ("0", "false", "")


def _capturing() -> bool:
    """CUDA-graph capture cannot sync or print; skip and report on a later step."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _mean(rows: list) -> torch.Tensor:
    return torch.stack(rows).float().mean(0)


def _fmt(values: list) -> str:
    cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
    return "[" + ", ".join(cells) + "]"


def _is_real_draft(spec, sample_pos) -> bool:
    """True on a serving step, false on the memory-profiling step.

    The profiling pass never runs input preparation, so ``sample_pos`` keeps
    its zero initialization; a real step places the block at ``P, P+1, ...``.
    """
    n_spec = spec.num_speculative_steps
    positions = [int(p) for p in sample_pos[0].tolist()]
    return positions[0] > 0 and positions == list(
        range(positions[0], positions[0] + n_spec)
    )


def _cache_gap(spec, idx_map, step, expected) -> torch.Tensor:
    """Max |cached - recomputed| logit of the batch's first row, or -1.

    The gumbel kernel caches the logits it sampled from, i.e. ``logits + bias``
    for the row it was given. A cache that differs from the bias recomputed
    here means the draft's own Markov chain did not see the token it should
    have, which is the one draft-side defect that only shows up from position 1
    on.
    """
    try:
        cache = spec.draft_logits
        if cache is None or idx_map.numel() == 0:
            return expected.new_tensor(-1.0)
        # idx_map is [num_reqs, n_spec]; the first request's row is enough.
        row = int(idx_map.reshape(-1)[0])
        if row < 0:
            return expected.new_tensor(-1.0)
        if cache.row_mapping is not None:
            row = int(cache.row_mapping[row])
            if row < 0:
                return expected.new_tensor(-1.0)
        col = int(spec._step_cols[step])
        cached = cache.values[row, col].float()
        return (cached - expected.float()).abs().max()
    except Exception:  # noqa: BLE001 - the buffer layout is not a patch API
        return expected.new_tensor(-1.0)


def _block_input(spec) -> str:
    """Tokens and positions the draft backbone read for the first block.

    Rows of one request are contiguous and the anchor is row 0, so the second
    row on must be the mask token repeated at ``P, P+1, ...``. Anything else
    (all rows the anchor's id, a zeroed mask, or a shifted position) makes
    positions 1..n worse while leaving the first draft token intact.
    """
    try:
        buf = spec.input_buffers
        n = int(spec._anchor_idx[1]) if spec._anchor_idx.numel() > 1 else 8
        ids = [int(v) for v in buf.input_ids[:n].tolist()]
        pos = [int(v) for v in buf.positions[:n].tolist()]
        # prepare_dflash_inputs writes the block's query slots into the shared
        # BlockTables buffer, which is what the draft attention reads.
        slots = "?"
        try:
            group = int(spec.draft_kv_cache_group_id)
            table = spec.block_tables.slot_mappings[group]
            slots = [int(v) for v in table[:n].tolist()]
        except Exception:  # noqa: BLE001 - optional detail
            pass
        return (
            f"mask={spec.parallel_drafting_token_id}"
            f" n_query={spec.num_query_per_req}"
            f" ids={ids} positions={pos} slots={slots}"
        )
    except Exception:  # noqa: BLE001 - the buffer layout is not a patch API
        return "unavailable"


def _accumulate_draft(
    spec, base_logits, hidden, prev, idx_map, sample_pos, out_tokens
) -> None:
    n_spec = spec.num_speculative_steps
    rows = int(out_tokens.shape[0])
    by_step = hidden.view(rows, n_spec, -1)
    acc = _draft_acc
    if not acc:
        acc.update(
            count=0,
            cos=[],
            bias=[],
            base=[],
            flip=[],
            match=[],
            top1=[],
            tok=[],
            tokflip=[],
            gap=[],
            anchor=int(prev[0]),
            block=_block_input(spec),
            sample_pos=[int(p) for p in sample_pos[0].tolist()],
            tokens=[int(t) for t in out_tokens[0].tolist()],
            seq_lens=[int(v) for v in spec.input_buffers.seq_lens[:2].tolist()],
        )
    acc["cos"].append(
        torch.stack(
            [
                F.cosine_similarity(
                    by_step[0, i].float(), by_step[0, i + 1].float(), dim=0
                )
                for i in range(n_spec - 1)
            ]
        )
    )
    # Padded rows (idx_map == -1) are skipped by the sampler and keep the
    # previous step's tokens, so their statistics would be stale rather than
    # random. Average over the live rows only.
    live = (idx_map[:, 0] >= 0).float()
    denom = live.sum().clamp(min=1)

    def row_mean(values: torch.Tensor) -> torch.Tensor:
        return (values.float() * live).sum() / denom

    def row_rms(values: torch.Tensor) -> torch.Tensor:
        cells = denom * values.shape[-1]
        return ((values.float() * live.unsqueeze(-1)).pow(2).sum() / cells).sqrt()

    if spec._draft_topk is None:
        state = None
        cur = prev
        per_step = []
        gap = []
        for i in range(n_spec):
            embed, state = spec._markov_step(cur, by_step[:, i], state)
            bias = spec.model.markov_bias(embed)
            base = base_logits[:, i]
            corrected = base + bias
            per_step.append(
                torch.stack(
                    (
                        row_rms(bias),
                        row_rms(base),
                        row_mean(base.argmax(-1) != corrected.argmax(-1)),
                        row_mean(corrected.argmax(-1) == out_tokens[:, i]),
                        row_mean(corrected.float().softmax(-1).max(-1).values),
                    )
                )
            )
            gap.append(_cache_gap(spec, idx_map, i, corrected[0]))
            # Swap the token the chain is conditioned on: a bias that does not
            # move has no token signal to carry into the next position.
            alt_embed, _ = spec._markov_step(
                (cur + 1) % spec.vocab_size, by_step[:, i], None
            )
            alt = spec.model.markov_bias(alt_embed)
            per_step[-1] = torch.cat(
                (
                    per_step[-1],
                    torch.stack(
                        (
                            row_rms(bias - alt) / row_rms(bias).clamp(min=1e-6),
                            row_mean(
                                corrected.argmax(-1) != (base + alt).argmax(-1)
                            ),
                        )
                    ),
                )
            )
            cur = out_tokens[:, i]
        stats = torch.stack(per_step)
        acc["bias"].append(stats[:, 0])
        acc["base"].append(stats[:, 1])
        acc["flip"].append(stats[:, 2])
        acc["match"].append(stats[:, 3])
        acc["top1"].append(stats[:, 4])
        acc["tok"].append(stats[:, 5])
        acc["tokflip"].append(stats[:, 6])
        acc["gap"].append(torch.stack(gap))
    acc["count"] += 1
    if acc["count"] >= _STEPS:
        _reported.add("draft")
        _flush_draft(spec)


def _flush_draft(spec) -> None:
    acc = _draft_acc
    n_spec = spec.num_speculative_steps
    lines = [
        f"DSpark draft ({acc['count']} real steps) n_spec={n_spec} "
        f"fuse_bias={spec._fuse_markov_bias} draft_topk={spec._draft_topk}",
        f"  anchor(req0)={acc['anchor']} seq_lens[:2]={acc['seq_lens']}",
        f"  backbone input(req0) {acc['block']}",
        f"  sample_pos(req0)={acc['sample_pos']}",
        f"  draft tokens(req0)={acc['tokens']}",
        f"  backbone hidden cos(req0, adjacent)={_fmt(_mean(acc['cos']).tolist())}",
    ]
    if acc["top1"]:
        columns = {
            "bias_rms": acc["bias"],
            "base_rms": acc["base"],
            "argmax_flip": acc["flip"],
            "argmax==draft": acc["match"],
            "draft_top1": acc["top1"],
            "bias_tok_gap": acc["tok"],
            "bias_tok_flip": acc["tokflip"],
            "cache_gap": acc["gap"],
        }
        means = {key: _mean(value) for key, value in columns.items()}
        for i in range(n_spec):
            cells = " ".join(f"{k}={float(v[i]):.4f}" for k, v in means.items())
            lines.append(f"  step {i}: {cells}")
    logger.warning("DSpark acceptance diagnostics (draft)\n%s", "\n".join(lines))


def _verify_rows(cu_num_logits, seq_lens_upper_bound_np) -> int:
    """Rows per request of a serving step, or 0 for the profiling step."""
    rows = int(cu_num_logits[1] - cu_num_logits[0])
    if rows <= 1:
        return 0
    # The profiling pass runs a synthetic 17-token batch; benchmark prompts
    # (gsm8k) are far longer than this.
    if min(int(v) for v in seq_lens_upper_bound_np.tolist()) <= 32:
        return 0
    return rows


def _target_dist(processed_logits, temperature) -> torch.Tensor:
    """Per-row target distribution, from the logits the sampler consumes."""
    logits = processed_logits.float()
    dist = logits.softmax(-1)
    greedy = temperature <= 0
    if bool(greedy.any()):
        one_hot = F.one_hot(logits.argmax(-1), dist.shape[-1]).float()
        dist = torch.where(greedy.unsqueeze(-1), one_hot, dist)
    return dist


def _draft_dist(draft_logits, expanded_idx_mapping, expanded_local_pos, n_spec):
    """Per-row draft distribution the ratio test divides by, and its validity.

    The gumbel kernel caches the logits it sampled from *before* temperature,
    at the row the draft step wrote and at the column of the row's local
    position, so this reads exactly what the sampler compares against. A
    sharded cache only holds the requests this rank verifies, so rows owned by
    another rank are reported invalid instead of read.
    """
    req_state = expanded_idx_mapping.clamp(min=0)
    rows = req_state
    valid = torch.ones_like(req_state, dtype=torch.bool)
    if draft_logits.row_mapping is not None:
        rows = draft_logits.row_mapping[req_state]
        valid = rows >= 0
        rows = rows.clamp(min=0)
    cols = expanded_local_pos.long().clamp(min=0, max=n_spec - 1)
    return draft_logits.values[rows.long(), cols].float(), valid


def _accumulate_verify(
    sampler,
    processed_logits,
    draft_logits,
    draft_sampled,
    num_sampled,
    pos,
    cu_num_logits,
    expanded_idx_mapping,
    expanded_local_pos,
    seq_lens_upper_bound_np,
) -> None:
    rows = _verify_rows(cu_num_logits, seq_lens_upper_bound_np)
    if rows == 0 or draft_logits is None:
        return
    n_spec = sampler.num_speculative_steps
    num_reqs = int(cu_num_logits.shape[0] - 1)
    # A few requests per step is enough for the per-position trend and keeps
    # the per-step vocabulary-sized temporaries small.
    n_reqs = max(1, min(num_reqs, 8))
    # The expanded arrays are sized by the batch, not by `num_reqs`, so the
    # request count has to fit them in whole rows.
    available = min(
        int(expanded_idx_mapping.numel()),
        int(processed_logits.shape[0]),
        int(draft_sampled.numel()),
    )
    n_reqs = min(n_reqs, available // rows)
    if n_reqs == 0:
        return
    take = rows * n_reqs
    req_state = expanded_idx_mapping[:take]
    temperature = sampler.sampler.sampling_states.temperature.gpu[
        req_state.clamp(min=0)
    ].float()
    # ``processed_logits`` is [num_logits, vocab] (or [num_logits, 1, vocab]).
    target = _target_dist(processed_logits[:take].flatten(0, -2), temperature)
    cache, valid = _draft_dist(
        draft_logits, req_state, expanded_local_pos[:take], n_spec
    )
    draft = (cache / temperature.clamp(min=1e-6).unsqueeze(-1)).softmax(-1)
    # Row r holds the distribution that sampled the token of row r + 1, so rows
    # 0 .. n_spec-1 are the draft positions and the last row is the bonus.
    target = target.view(n_reqs, rows, -1)
    draft = draft.view(n_reqs, rows, -1)
    draft_sampled = draft_sampled[:take].view(n_reqs, rows)
    # Row r holds the distribution that sampled the token of row r + 1, so the
    # draft positions are rows 0 .. n_pos-1 and the last row is the bonus.
    # Adaptive verification may verify fewer than n_spec positions.
    n_pos = max(0, min(n_spec, rows - 1))
    if n_pos == 0:
        return
    weight = valid.view(n_reqs, rows).float()[:, :n_pos]
    best = target.argmax(-1)
    acc = _verify_acc
    if not acc:
        acc.update(
            count=0,
            rows=rows,
            n_pos=n_pos,
            num_reqs=num_reqs,
            temperature=[float(v) for v in temperature[:2].tolist()],
            locals=[int(p) for p in expanded_local_pos[:rows].tolist()],
            positions=[int(p) for p in pos[:rows].tolist()],
            seq_lens=[int(v) for v in seq_lens_upper_bound_np[:n_reqs].tolist()],
            draft=[int(t) for t in draft_sampled[0].tolist()],
            cached=[] if draft_logits.row_mapping is not None else None,
            shape=list(draft_logits.values.shape),
            accepted=[],
            top1=[],
            draft_top1=[],
            q_at_best=[],
            overlap=[],
            greedy=[],
            argmax_agreement=[],
            base_overlap=[],
            tf_overlap=[],
            base_argmax=[],
            tf_argmax=[],
        )

    def per_position(values):
        """Mean over the sampled requests, skipping rows this rank cannot see."""
        cells = values[:, :n_pos] * weight
        return cells.sum(0) / weight.sum(0).clamp(min=1)

    acc["accepted"].append(float(num_sampled[:n_reqs].float().mean()))
    acc["top1"].append(per_position(target.max(-1).values))
    acc["draft_top1"].append(per_position(draft.max(-1).values))
    acc["q_at_best"].append(
        per_position(draft.gather(-1, best.unsqueeze(-1)).squeeze(-1))
    )
    acc["overlap"].append(per_position(torch.minimum(target, draft).sum(-1)))
    acc["greedy"].append(
        per_position((best[:, :n_pos] == draft_sampled[:, 1 : n_pos + 1]).float())
    )
    # What a greedy draft would buy: two argmaxes agree exactly when the
    # sampler accepts its own draw, so this is the per-position acceptance a
    # temperature-0 run would report.
    acc["argmax_agreement"].append(
        per_position((best[:, :n_pos] == draft[:, :n_pos].argmax(-1)).float())
    )
    # Split the depth loss in two. A cached row is ``base + bias(token used by
    # the chain)``, so subtracting that bias gives the backbone's own
    # distribution, and adding the bias of the token the target itself would
    # have used gives the teacher-forced one. Neither needs another forward.
    if n_pos == acc["n_pos"] and _draft_model[0] is not None:
        try:
            model = _draft_model[0]
            raw = cache.view(n_reqs, rows, -1)[0]
            t0 = temperature[0].clamp(min=1e-6)
            base_ov, tf_ov, base_am, tf_am = [], [], [], []
            for i in range(1, n_pos):
                if not bool(valid[0, i]):
                    break
                # Draft ids index the same vocabulary as the target when the
                # draft is full-vocabulary (draft_vocab_size == vocab_size).
                used = draft_sampled[0, i - 1 : i]
                base = raw[i] - model.markov_bias(model.markov_embed(used))[0].float()
                base_p = (base / t0).softmax(-1)
                true_tok = best[0, i - 1 : i]
                tf_p = (
                    (
                        base
                        + model.markov_bias(model.markov_embed(true_tok))[0].float()
                    )
                    / t0
                ).softmax(-1)
                q = target[0, i]
                base_ov.append(torch.minimum(base_p, q).sum())
                tf_ov.append(torch.minimum(tf_p, q).sum())
                base_am.append((base_p.argmax() == best[0, i]).float())
                tf_am.append((tf_p.argmax() == best[0, i]).float())
            if len(base_ov) == n_pos - 1:
                acc["base_overlap"].append(torch.stack(base_ov))
                acc["tf_overlap"].append(torch.stack(tf_ov))
                acc["base_argmax"].append(torch.stack(base_am))
                acc["tf_argmax"].append(torch.stack(tf_am))
        except Exception:  # noqa: BLE001 - the draft head is not a patch API
            pass
    if acc["cached"] is not None:
        acc["cached"].append(int(valid.sum()))
    acc["count"] += 1
    if acc["count"] >= _STEPS:
        _reported.add("verify")
        _flush_verify(sampler)


def _flush_verify(sampler) -> None:
    acc = _verify_acc
    positions = acc["positions"]
    keys = (
        "top1",
        "draft_top1",
        "q_at_best",
        "overlap",
        "greedy",
        "argmax_agreement",
        "base_overlap",
        "tf_overlap",
        "base_argmax",
        "tf_argmax",
    )
    means = {key: _mean(acc[key]).tolist() for key in keys if acc[key]}
    accepted = sum(acc["accepted"]) / len(acc["accepted"])
    if acc["cached"] is None:
        cached = f"replicated shape={acc['shape']}"
    else:
        owned = sum(acc["cached"]) / len(acc["cached"])
        cached = f"{owned:.2f} of {acc['rows']} sharded shape={acc['shape']}"
    lines = [
        f"DSpark verify ({acc['count']} serving steps) rows_per_req={acc['rows']}"
        f" n_pos={acc['n_pos']}"
        f" num_reqs={acc['num_reqs']} n_spec={sampler.num_speculative_steps}"
        f" block_verification={sampler.use_block_verification}",
        f"  temperature={acc['temperature']} local_pos(req0)={acc['locals']}",
        f"  markov bias scale={_BIAS_SCALE:.3f}"
        f" ({_BIAS_SCALE_ENV}; 0=plain backbone, 1=shipped)",
        f"  positions(req0)={positions} deltas="
        f"{[p - positions[0] for p in positions]} seq_lens={acc['seq_lens']}",
        f"  draft tokens(req0)={acc['draft']}",
        f"  accepted tokens per block={accepted:.4f} cache rows={cached}",
        f"  target top1 per position={_fmt(means['top1'])}",
        f"  draft top1 per position={_fmt(means['draft_top1'])}",
        f"  draft mass at target argmax={_fmt(means['q_at_best'])}",
        f"  sum min(p, q) per position={_fmt(means['overlap'])}",
        f"  argmax(target) == draft token={_fmt(means['greedy'])}",
        f"  argmax(target) == argmax(draft)={_fmt(means['argmax_agreement'])}",
        f"  backbone only, rows 1..{acc['n_pos'] - 1}: "
        f"sum min(base, q)={_fmt(means.get('base_overlap', []))}"
        f" argmax==target={_fmt(means.get('base_argmax', []))}",
        f"  teacher-forced prev token, rows 1..{acc['n_pos'] - 1}: "
        f"sum min(p, q)={_fmt(means.get('tf_overlap', []))}"
        f" argmax==target={_fmt(means.get('tf_argmax', []))}",
    ]
    logger.warning("DSpark acceptance diagnostics (verify)\n%s", "\n".join(lines))


def _install() -> None:
    global _BIAS_SCALE
    if not _enabled() or not current_platform.is_ppu():
        return
    _BIAS_SCALE = _resolve_bias_scale()
    if _BIAS_SCALE != 1.0:
        logger.warning(
            "DSpark diagnostics: Markov bias scaled by %.3f (%s); acceptance "
            "leaving the shipped behaviour.",
            _BIAS_SCALE,
            _BIAS_SCALE_ENV,
        )
    try:
        from vllm.v1.worker.gpu.spec_decode import rejection_sampler
        from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break loading
        logger.warning("DSpark acceptance diagnostics not installed: %r", exc)
        return

    upstream_chain = dspark.DSparkSpeculator._sample_markov_chain
    upstream_verify = rejection_sampler.RejectionSampler._verify

    def sample_markov_chain(
        self, base_logits, hidden, prev, idx_map, sample_pos, out_tokens
    ):
        # Before the upstream call: it reads the bias from this model.
        if _BIAS_SCALE != 1.0 and not _capturing():
            _install_bias_scale(self.model, _BIAS_SCALE)
        confidence = upstream_chain(
            self, base_logits, hidden, prev, idx_map, sample_pos, out_tokens
        )
        if "draft" not in _reported and not _capturing():
            _draft_model[0] = self.model
            try:
                if _is_real_draft(self, sample_pos):
                    _accumulate_draft(
                        self,
                        base_logits,
                        hidden,
                        prev,
                        idx_map,
                        sample_pos,
                        out_tokens,
                    )
            except Exception as exc:  # noqa: BLE001
                # Start the window over: a raise in the middle of a step would
                # leave the per-position lists of different lengths.
                _draft_acc.clear()
                _failures["draft"] = _failures.get("draft", 0) + 1
                if _failures["draft"] >= _MAX_FAILURES:
                    _reported.add("draft")
                logger.warning(
                    "DSpark draft diagnostics failed (%d/%d): %r",
                    _failures["draft"],
                    _MAX_FAILURES,
                    exc,
                )
        return confidence

    def verify(
        self,
        logits,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
        seq_lens_upper_bound_np,
    ):
        result = upstream_verify(
            self,
            logits,
            draft_logits,
            draft_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            idx_mapping_np,
            expanded_idx_mapping,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        if "verify" not in _reported:
            try:
                processed_logits, _, num_sampled = result
                _accumulate_verify(
                    self,
                    processed_logits,
                    draft_logits,
                    draft_sampled,
                    num_sampled,
                    pos,
                    cu_num_logits,
                    expanded_idx_mapping,
                    expanded_local_pos,
                    seq_lens_upper_bound_np,
                )
            except Exception as exc:  # noqa: BLE001
                _verify_acc.clear()
                _failures["verify"] = _failures.get("verify", 0) + 1
                if _failures["verify"] >= _MAX_FAILURES:
                    _reported.add("verify")
                logger.warning(
                    "DSpark verify diagnostics failed (%d/%d): %r",
                    _failures["verify"],
                    _MAX_FAILURES,
                    exc,
                )
        return result

    patch(
        _DSPARK_MODULE,
        "DSparkSpeculator._sample_markov_chain",
        reason=(
            "TEMPORARY diagnostic: report the draft block's positions, the "
            "backbone hidden similarity across positions and the Markov bias "
            "magnitude/flip rate once, to locate why DSpark acceptance decays "
            "faster than the reference per-position curve."
        ),
        affected_versions=_AFFECTED,
        remove_when="The acceptance gap is explained; delete this module.",
    )(sample_markov_chain)

    patch(
        _REJECTION_MODULE,
        "RejectionSampler._verify",
        reason=(
            "TEMPORARY diagnostic: compare the target distribution the sampler "
            "consumed with the cached draft distribution it divides by, per "
            "verified position, so a draft that is worse than the reference can "
            "be told from a verify step that under-accepts its own inputs."
        ),
        affected_versions=_AFFECTED,
        remove_when="The acceptance gap is explained; delete this module.",
    )(verify)


_install()
