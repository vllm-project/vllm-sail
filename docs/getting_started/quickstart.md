# Quickstart

Complete [installation](installation.md) and its native extension checks first.
Choose a checkpoint whose architecture, precision and backend are covered by
the [feature matrix](../user_guide/supported_features.md), and which fits your
available PPU memory. Replace `/path/to/model` with the local checkpoint path.

## Online serving

```bash
vllm serve /path/to/model \
  --served-model-name ppu-model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --host 127.0.0.1 --port 8000
```

Wait for the server to finish loading, then query it from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ppu-model","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

The health endpoint checks readiness. A completion exercises model execution;
check the returned text as well as the HTTP status. Chat endpoints additionally
require an appropriate chat template for the checkpoint.

## Offline inference

Save the following as `inference.py` and run `python inference.py` in the same
environment:

```python
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/path/to/model",
        dtype="bfloat16",
        max_model_len=4096,
    )
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=16),
    )
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
```

vLLM loads the installed plugin automatically. Applications use the standard
vLLM API; no application-level registration call is needed.

## Configure your workload

- **Multiple devices:** set `--tensor-parallel-size` to the intended device
  count, after confirming the model and communication libraries support it.
- **Quantization:** use a checkpoint with a supported quantization format, or
  loading-time conversion (`fp8-to-int8`, `bf16-to-int8`, `fp8-to-bf16`; see
  [Online quantization](../user_guide/online_quantization.md)). Match the
  model's metadata and the backend constraints in the feature matrix.
- **Backend selection:** use the defaults first; the
  [configuration guide](../user_guide/configuration.md) describes explicit
  DeepGEMM, ACEXT and Triton selection.
- **Context length:** adjust `--max-model-len` to the model and available memory.

For correctness tests and performance reporting, follow the
[verification guide](../user_guide/verification_guide.md).
