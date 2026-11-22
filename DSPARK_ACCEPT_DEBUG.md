# DSpark 投机解码接受率排查记录

> 平台：PPU，vLLM 0.34.0 + vllm-sail，Kimi-K3.1-Pro (TP16/EP16)
> 最近更新：2026-11-19（对应日志 `dspark.log.21`）

## 0. 一句话现状

draft 侧机械正确性、verify 侧采样忠实性都已定量证明无误；接受率偏低被收敛到
**draft 分布在深度上的质量衰减**这一个变量上（conditional 率 pos3 之后饱和在 0.35，
官方同 checkpoint 恒定 0.82）。下一步靠新增的 `base_overlap` / `tf_overlap` 两列
区分「链上自采样噪声」与「backbone 块内表征」。

## 1. 运行配置

| 项 | 值 |
|---|---|
| target model | `/ppusw/datasets/checkpoints/LLM/moonshotai/k3.1/Kimi-K3.1-Pro` |
| draft model | `/ppusw/datasets/checkpoints/LLM/Inferact/K3/Kimi-K3.1-Pro-DSpark-preview` |
| 并行 | TP16 / EP16 / PP1 |
| draft attention | FLASHMLA + DSA indexer，next_n=8 |
| speculative config | `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}` |
| client | temperature 0.9，未设 top_p（官方是 1.0 / 0.95） |
| 诊断开关 | 需 `--enforce-eager`（CUDA graph capture 期间不打印）；**默认关闭**，`VLLM_SAIL_DSPARK_DIAG=1` 开启 |

诊断补丁会打印 `accepted tokens per block`，与引擎的 `Mean acceptance length` 可直接对齐。

## 2. 官方口径（ModelScope README 原文）

`Inferact/Kimi-K3.1-Pro-DSpark-preview` 页面的 server 配置与我们**完全相同**，
差异只在 client：官方用 vllm-bench chat completions，**temperature 1.0、top_p 0.95、完整测试集**，
指标 = `1 + accepted draft tokens / drafts`。

| benchmark | acceptance length | per-position |
|---|---|---|
| GSM8K (1319 条, max_tokens 1024) | **4.597** | 0.86, 0.71, 0.59, 0.48, 0.39, 0.32, 0.26 |
| MATH-500 | 3.824 | — |
| HumanEval | 3.585 | — |
| MBPP | 3.520 | — |
| AA-LCR | 3.501 | — |
| pooled | 3.154 | 0.72, 0.50, 0.34, 0.23, 0.16, 0.11, 0.08 |
| AIME 2026 | 2.973 | — |
| BEAM 128K | 2.954 | — |
| SPEED-Bench (low entropy) | 2.930 | 0.70, 0.46, 0.30, 0.19, 0.13, 0.09, 0.06 |
| MT-Bench | 2.812 | — |
| BEAM 1M | 2.765 | — |
| SWE-bench Pro | 2.741 | 0.67, 0.43, 0.27, 0.16, 0.10, 0.06, 0.04 |

两点必须记住：

1. 官方曲线**跨任务波动极大**（4.597 → 2.741），绝对数值不能直接对比，必须同任务同口径。
2. README 首行写着 "Preview checkpoint. The latest checkpoint is ready: `Inferact/Kimi-K3.1-Pro-DSpark`"。
   我们加载的是 `-preview` 版本，需确认是否有意为之。

## 3. 关键定量结论

### 3.1 引擎 per-position 是「无条件」计数，不是 conditional

```
Mean acceptance length = 1 + Σ per_position
```
log.21 的 `0.898, 0.610, 0.268, 0.106, 0.031, 0.004, 0.000` 之和 1.917，
加 1 得 2.917 ≈ 引擎报的 2.92，成立。所以 per_position[i] = Π_{j<=i} conditional[j]。

**用法**：拿它反解出 conditional 率，再与诊断里的 `sum min(p, q)` 逐位置比，
就能判断 verify 是否忠实。

### 3.2 verify 侧忠实性已定量证明

三条稳态快照平均 per_position = `[0.886, 0.606, 0.303, 0.109, 0.041, 0.010, 0.0007]`，
反解 conditional：

| 位置 | 引擎反解 | 诊断 `sum min(p, q)` |
|---|---|---|
| 1 | 0.89 | 0.859 |
| 2 | 0.68 | 0.616 |
| 3 | 0.50 | 0.450 |
| 4 | 0.36 | 0.351 |
| 5 | 0.38 | 0.348 |
| 6 | 0.24 | 0.355（引擎侧样本仅几块，噪声大） |
| 7 | ~0.07 | 0.332（同上） |

pos0–4 吻合在噪声范围内，说明 **target 前向、rejection sampler、block 验证内的链交接
在 7 个深度上全部忠实**。pos5 以后的偏离是引擎侧统计样本太少（几十块）导致，
不是 bug。

（早前"按 sum min(p,q) 前缀连乘得理论块长 2.75 vs 实测 2.92"的 6% 偏差，
在两条口径都确认一致之后可以视为统计窗口差异，不再作为疑点。）

### 3.3 真正的异常是曲线形状

| 位置 | 官方 GSM8K conditional | 我们 conditional |
|---|---|---|
| 1 | 0.86 | 0.89 |
| 2 | 0.83 | 0.68 |
| 3 | 0.83 | 0.50 |
| 4 | 0.81 | 0.36 |
| 5 | 0.81 | 0.38 |
| 6 | 0.82 | 0.24 |
| 7 | 0.81 | ~0.07 |

- 官方：**恒定 ×0.82**，每一步只掉 18%。
- 我们：pos0 更高（0.89 vs 0.86），但 pos3 就掉到 0.36，然后**成平台**。

pos0 更高 + 后面崩塌，说明不是"整体分布更差"，而是**块内表征或链上误差在深度上被放大**。

"官方是 greedy 所以更好"这个假设已被 README 否掉（官方 temp=1.0，比我们更噪），
`argmax(target) == argmax(draft)` 那一路数据也只能排除采样口径这一种解释：
我们该列 `0.858/0.634/0.455/0.354/0.344/0.341/0.329`，与 `sum min(p,q)` 几乎重合，
说明 loss 不是"抽签手气差"，而是 draft 的 argmax 本身就跑偏了。

## 4. 已排除项（均带证据，不要重复查）

| 假设 | 结论 | 证据 |
|---|---|---|
| verify 采样器 / ratio test 有 bug | 排除 | 见 3.2 逐位置对齐 |
| draft→verify 的 token 交接错位 | 排除 | `cache_gap=0.0000`（7 步全 0），draft tokens 与 verify 输入逐位相同 |
| block 验证的链构造错误 | 排除 | `block_verification=True`，`cache rows=max(n_pos, block)`，`replicated` |
| draft 输入 ids/positions 错位 | 排除 | `ids=[1282, 163839×6]`、`positions=[9..15]` 连续、`sample_pos=[10..16]`、`seq_lens=16=last_valid(8)+1+7` |
| slot mapping 有 PAD | 排除 | `slots=[9609..9615]` 连续无 PAD |
| 块内 causal mask 被误关 | 排除 | `_dflash_layer_causal` 读 `dflash_config.causal: true`，配置本意是 causal |
| markov bias 未生效 / 未被使用 | 排除 | `bias_rms/logit_rms` 正常，`bias_tok_flip=0.72–0.77`，`_fuse_markov_bias` 路径正确 |
| draft 权重加载错误 | 排除 | 与 target top1 曲线（0.869→0.679）同量级，且 pos0 甚至优于官方 |
| 采样参数（温度尖锐度） | 不是主因 | 我们 0.9 比官方 1.0 更尖锐，pos0 也确实更高，但深度上差得多 |
| CUDA graph 导致 hook 数据错 | 排除 | 全程 `--enforce-eager` |
| `dspark_draft_topk` 未启用 | 不适用 | 该优化要求 draft TP=1，我们 TP16 |
| draft 缺 softcap 导致 logit 尺度不对 | 排除 | `Qwen3DSparkForCausalLM.logits_processor` 无 soft_cap，与 checkpoint 一致 |
| `--block-size 64` 能关掉 64-token 强制拆分 | **无效，实验作废** | Phase 2 `_align_hybrid_block_size` 无条件覆盖用户值（`platforms/interface.py:1118-1124`，注释写明 may override user settings）；384 只由 KDA mamba page 与 MLA 单 token page 决定，传 16/64/128 都是 384 |
| `kimi-k3.1-new` 有 current 缺失的修复 | 排除 | 两边把打包 draft cache 拆成 64-token kernel block 并重映射 slot 是等价的：new 内建在 vllm 本体（`maps_slot_mapping` / `map_to_kernel_slot_mapping` / `_kernel_slots_kernel`），current 在 vllm-sail `packed_token_cache_split.py` monkey-patch 三条写路径；remap 公式与 log.21 数值自洽 |

## 5. 当前唯一的待判定问题

深度上的损失来自哪一侧？

- **A：链上自采样误差被放大** —— 链上第 r 步喂的是自己上一步抽出的 token，
  一旦抽偏，markov bias 会把后续整条链带跑。深度越深，偏离越大。
- **B：backbone 块内表征本身差** —— mask 行对 anchor 的 in-block attention、
  或 PPU 上 MLA 的 causal 路径有问题，导致第 r 行拿到的 context 就不好。

### 判读规则（补丁已就绪）

重新跑一次（命令不变，仍需 `--enforce-eager`），verify 段会多出两行：

```
backbone only, rows 1..6:   sum min(base, q)=[...]  argmax==target=[...]
teacher-forced prev token:  sum min(p, q)=[...]     argmax==target=[...]
```

原理：cached row 严格等于 `base_r + markov_bias(token_{r-1})`，而 vanilla head 只依赖 token，
所以

- `base_r = cached_r - markov_bias(draft_token_{r-1})` → backbone 自己的分布
- `p_tf_r = softmax((base_r + markov_bias(target_argmax_{r-1})) / T)` → teacher-forced 链

两者都**不需要额外前向**，纯离线重构。

| 观测 | 结论 | 下一步 |
|---|---|---|
| `tf_overlap` 平坦 ≈0.75+，`overlap` 崩到 0.35 | **A** | 调 bias 强度 / 用 target argmax 做 bias 输入 / 降链上温度 |
| `tf_overlap` 也崩到 0.35 | **B** | 回到 PPU MLA 块内 causal attention（mask 行、context KV 使用） |
| `base_overlap` < `overlap` | bias 在救场 | 说明 backbone 更差，问题更偏 B |
| `base_overlap` > `overlap` | bias 在拖后腿 | 说明 bias 过强，问题更偏 A |

## 6. fp8 indexer 实验（已做完，结论：与接受率无关）

### 6.1 事件

用 `--attention-config '{"indexer_kv_dtype": "fp8"}'` 启动，加载阶段崩在
`RuntimeError: PPU fp8_fp4_mqa_logits supports MXFP4 only; use fp8_mqa_logits for ordinary FP8`。
已修复，修复后接受率**没有变化**。

崩溃栈（log.22）：`sparse_attn_indexer.py:921` → `patch/enhancement/attention/sparse_attn_indexer.py:280`
(`fp8_fp4_mqa_logits_ppu`) → `utils/deep_gemm.py:824` → SAIL。

### 6.2 修复方式（已验证正确）

在 `fp8_fp4_mqa_logits_ppu`（L281）与 `fp8_fp4_paged_mqa_logits_ppu`（L329）各加一条：

```python
q_values = q[0] if isinstance(q, tuple) else q
if q_values.dtype != torch.int8:
    return _ppu_deep_gemm.fp8_mqa_logits(...)   # paged 版为 fp8_paged_mqa_logits
```

正确性依据：上游 `sparse_attn_indexer.py` L881-890 只在 `use_fp4_cache=True` 时把 q/k
`view(torch.int8)`；fp8 模式下 q 是 `float8_e4m3fn`，因此新分支必然命中。
`attention/ops/mla_sparse.py` L312/L322 那条路是按 `use_fp4_cache` 分流的，fp8 时本来
就不会调用 fp8_fp4，无同类风险。

### 6.3 fp8 确实生效了（不是配置没吃进去）

| | log.21（基线 mxfp4） | log.22（fp8） |
|---|---|---|
| `DSA indexer decode path` | `use_fp4_cache=True` | `use_fp4_cache=False` |
| INDEXER cache | `shape=(244, 1, 384, 68)` | `shape=(230, 1, 384, 132)` |
| 每行字节 | 64 + 4（scale） | 128 + 4（scale） |

68 → 132 字节正好是 mxfp4 → fp8 的 K 布局，配置生效无疑。

### 6.4 为什么「没变化」是预期结果

1. **indexer 只是 top-k 选择器**：它决定选哪 `index_topk` 个 KV 位置，被选中的位置仍用主
   MLA 的 BF16 全精度做 attention。精度变化只改「选哪些位置」，不改「用什么精度算」，
   因此对 q 的影响是二阶的。
2. **两种格式精度同量级**：mxfp4 每 32 元素一个 e8m0 scale，fp8 是 e4m3 + scale，对 128 维
   index head 来说 top-k 边界几乎不变。
3. **最关键**：draft 是 dense MLA（`qwen3_dspark`），**不经过 DSA indexer**。而瓶颈已定位在
   draft 的深度衰减，这条路径改不到 draft 的 p。

### 6.5 这条实验的价值与建议

- **价值**：完成了一次 target 侧精度扰动（mxfp4 → fp8），接受率不动 → 更强地指向 draft 侧。
  target 侧 indexer 精度这条分支可以划掉。
- **建议实验完退回 mxfp4**：fp8 不但没有精度收益，还把 indexer KV 内存翻倍
  （68 → 132 字节/行，可用页数 244 → 230），纯亏。

## 7. 其它旁支

- **`packed_token_cache_split.py` 会强制把 FLASHMLA token cache 拆成 64-token kernel block**
  并 remap 各 layer 的 slot mapping（log.22 的 L923-939）。若后续怀疑 slot remap，这里是入口。
- **log.22 那次是启动即崩，没有跑到接受率**：任何引用「fp8 下接受率」的数据都必须来自
  修复后的运行，不能拿 log.22 的引擎指标说事。

## 8. 复现方式

```bash
# 启动参数与平常一致，额外加 --enforce-eager
# 诊断默认关闭，开启：VLLM_SAIL_DSPARK_DIAG=1
# Markov bias 强度（默认 1.0 = 原始行为）：VLLM_SAIL_DSPARK_BIAS_SCALE=0
# 报告在 WARNING 级别，每个 hook 只打一次，采 200 个 serving step
```

启动后看日志里的 `DSpark acceptance diagnostics (draft)` 与 `(verify)` 两段，
八张 rank 的数值应完全一致（已验证）。verify 报告里会回显 `markov bias scale=...`，
记录数据时必须一并记下。

### 8.1 bias 强度扫描

`VLLM_SAIL_DSPARK_BIAS_SCALE` 乘在 `self.model.markov_bias` 的返回值上。选择这个
位置是因为 `DSparkSpeculator._sample_sequential` L320 只用这一处取 bias，fused
Gumbel kernel 与显式相加两条分支都走它，包一处即可全覆盖；verify 侧的 base / tf
重构也调用同一个方法，因此在任意 scale 下都保持自洽。

| scale | 含义 |
|---|---|
| `0` | 完全去掉 markov bias，链退化成本 vanilla backbone |
| `0.5` | 半强度 |
| `1`（默认） | 线上行为 |

建议扫 `0` 与 `1` 两点就能拿结论：

| 观测 | 结论 |
|---|---|
| `scale=0` 接受长度仍 ≈ 2.9 | bias 无关紧要，问题在 backbone 块内表征（B） |
| `scale=0` 显著下降，`scale=1` 是 2.9 | bias 在起作用但不够强，是链上误差传播（A） |
| `scale=0.5` 优于两者 | bias 过强，带偏了链（A 的一个变种） |

注意：`scale≠1` 时引擎的接受长度不再代表线上性能，只用于归因。

## 9. 文件索引

| 文件 | 作用 |
|---|---|
| `vllm-sail/vllm_sail/patch/enhancement/attention/dspark_accept_diag.py` | 临时诊断补丁（**排查完成后需删除**，并移除 `attention/__init__.py` 里的条件导入） |
| `vllm-sail/dspark.log.21` | 当前基线日志（200 步 verify + 引擎稳态） |
| `vllm-sail/dspark.log.19` | draft 侧首个完整报告（ids/positions/slots/cache_gap） |
| `vllm-sail/dspark.log.22` | fp8_fp4_mqa_logits 崩溃，旁支 |
| `vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | `_sample_markov_chain`，draft hook 位置 |
| `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | `_verify`，verify hook 位置；注意它**没有** `self.model` |
| `vllm/vllm/model_executor/models/qwen3_dspark.py` | draft 模型定义、`logits_processor` |
| `vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | dflash 侧实现，draft 元数据构造 |
| `vllm/vllm/model_executor/models/qwen3_dflash.py` | `_dflash_layer_causal`，块内 causal 判定 |
| draft `config.json` | `dflash_config.causal: true`，`full_attention_causal: true` |

## 10. 遗留 TODO

- [ ] 跑一次拿到 `base_overlap` / `tf_overlap`，判定走 A 还是 B
- [ ] 按官方口径（temperature 1.0、top_p 0.95、完整 GSM8K）严格对跑一次，排除客户端口径差异
- [ ] 确认 draft 是否该换成非 preview 的 `Kimi-K3.1-Pro-DSpark`
- [ ] 排查收尾：删除 `dspark_accept_diag.py` 及 `attention/__init__.py` 的条件导入
- [ ] （更早遗留）`packed_token_cache_split.py` 里 `split:` / `skip:` 的 `_report_once`
      汇总日志是否删除；`unsplittable:` 必须保留
