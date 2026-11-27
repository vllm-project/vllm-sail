# SPDX-License-Identifier: Apache-2.0
"""Fuse CPU DP metadata coordination for target and first Step5 MTP pass."""

from contextvars import ContextVar
from functools import wraps

from vllm_sail import envs

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm.platforms import current_platform

    if not (envs.VLLM_SAIL_STEP5_FUSE_DP_COORDINATION and current_platform.is_ppu()):
        return

    import torch
    import torch.distributed as dist
    from vllm.v1.spec_decode import llm_base_proposer
    from vllm.v1.worker import dp_utils, gpu_model_runner

    from vllm_sail.patch.utils import patch

    metadata = dict(
        reason="Skip the separate first-pass Step5 MTP CPU DP collective.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="vLLM combines target and speculative DP coordination.",
    )
    target_context = ContextVar("step5_target_dp_coordination", default=None)
    draft_context = ContextVar("step5_draft_dp_coordination", default=None)
    coordinate = dp_utils.coordinate_batch_across_dp
    determine = gpu_model_runner.GPUModelRunner._determine_batch_execution_and_padding
    propose = llm_base_proposer.SpecDecodeBaseProposer.propose

    @patch(
        gpu_model_runner.__name__,
        "GPUModelRunner._determine_batch_execution_and_padding",
        **metadata,
    )
    @wraps(determine)
    def determine_target(self, *args, **kwargs):
        parallel = self.parallel_config
        spec = self.speculative_config
        if (
            parallel.data_parallel_size <= 1
            or parallel.pipeline_parallel_size != 1
            or not parallel.disable_nccl_for_dp_synchronization
            or parallel.use_ubatching
            or spec is None
            or spec.method != "mtp"
            or spec.draft_model_config is None
            or spec.draft_model_config.hf_config.model_type != "step5_mtp"
            or spec.disable_padded_drafter_batch
        ):
            return determine(self, *args, **kwargs)

        drafter = self.drafter
        drafter._sail_dp_first_pass = None
        if dp_utils.should_skip_dp_coordination():
            return determine(self, *args, **kwargs)
        # The normal execute_model call passes these by keyword. Capture and
        # dummy targets still use the same collective width, with no draft row.
        rows = kwargs.get("num_tokens", 0)
        requests = kwargs.get("num_reqs", 0)
        draft = None
        if (
            kwargs.get("force_uniform_decode") is None
            and not kwargs.get("force_eager", False)
            and self._is_uniform_decode(
                kwargs.get("max_num_scheduled_tokens"),
                self.uniform_decode_query_len,
                rows,
                requests,
            )
            and requests > 0
            and not drafter.needs_extra_input_slots
            and len(self.kv_cache_config.kv_cache_groups) > 0
            and int(self.optimistic_seq_lens_cpu[:requests].max().item())
            + self.num_spec_tokens
            <= self.effective_drafter_max_model_len
        ):
            mode, desc = drafter.cudagraph_dispatcher.dispatch(rows, valid_modes=None)
            draft = (rows, desc.num_tokens, mode.value)
        token = target_context.set((drafter, draft))
        try:
            return determine(self, *args, **kwargs)
        finally:
            target_context.reset(token)

    @patch(gpu_model_runner.__name__, "coordinate_batch_across_dp", **metadata)
    @wraps(coordinate)
    def coordinate_target(
        num_tokens_unpadded,
        allow_microbatching,
        parallel_config,
        num_tokens_padded=None,
        uniform_decode=None,
        cudagraph_mode=0,
    ):
        state = target_context.get()
        if state is None:
            return coordinate(
                num_tokens_unpadded,
                allow_microbatching,
                parallel_config,
                num_tokens_padded,
                uniform_decode,
                cudagraph_mode,
            )
        drafter, draft = state
        assert not (
            allow_microbatching
            and dp_utils.check_ubatch_thresholds(
                parallel_config, num_tokens_unpadded, uniform_decode=uniform_decode
            )
        )
        _, group = dp_utils._get_device_and_group(parallel_config)
        rank = parallel_config.data_parallel_rank
        # Rows 0:4 follow the native layout. Row 7 is an all-rank readiness bit.
        payload = torch.zeros(
            (8, parallel_config.data_parallel_size), dtype=torch.int32, device="cpu"
        )
        payload[0, rank] = num_tokens_unpadded
        payload[1, rank] = (
            num_tokens_unpadded if num_tokens_padded is None else num_tokens_padded
        )
        payload[3, rank] = cudagraph_mode
        if draft is not None:
            payload[4:8, rank] = torch.tensor(
                (*draft, 1), dtype=torch.int32, device="cpu"
            )
        dist.all_reduce(payload, group=group)

        if bool(payload[7].all()):
            assert draft is not None
            mode = int(payload[6].min())
            draft_meta = payload[4:8]
            drafter._sail_dp_first_pass = (
                draft,
                dp_utils._post_process_dp_padding(draft_meta, mode != 0),
                mode,
            )

        target_mode = dp_utils._post_process_cudagraph_mode(payload[:4])
        return (
            False,
            dp_utils._post_process_dp_padding(payload[:4], target_mode != 0),
            target_mode,
        )

    @patch(llm_base_proposer.__name__, "SpecDecodeBaseProposer.propose", **metadata)
    @wraps(propose)
    def propose_first_pass(self, *args, **kwargs):
        ticket = getattr(self, "_sail_dp_first_pass", None)
        self._sail_dp_first_pass = None
        token = draft_context.set(ticket)
        try:
            return propose(self, *args, **kwargs)
        finally:
            draft_context.reset(token)

    @patch(llm_base_proposer.__name__, "coordinate_batch_across_dp", **metadata)
    @wraps(coordinate)
    def coordinate_draft(
        num_tokens_unpadded,
        allow_microbatching,
        parallel_config,
        num_tokens_padded=None,
        uniform_decode=None,
        cudagraph_mode=0,
    ):
        ticket = draft_context.get()
        if ticket is not None:
            # Single consumption attempt per produce, match or not, so a stale
            # ticket is never reused after a fallback.
            draft_context.set(None)
            predicted, padding, mode = ticket
            # The fused target all_reduce already issued tickets to all ranks or
            # none, and the prediction is rank-invariant, so a mismatch falls
            # back on every rank without a second collective.
            actual = (num_tokens_unpadded, num_tokens_padded, cudagraph_mode)
            if predicted == actual:
                return False, padding, mode

        return coordinate(
            num_tokens_unpadded,
            allow_microbatching,
            parallel_config,
            num_tokens_padded,
            uniform_decode,
            cudagraph_mode,
        )

    _installed = True


install()
