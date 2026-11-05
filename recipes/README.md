# vLLM SAIL recipes

Zhenwu-specific recipe overlays live under `overlays/<org>/<model>.yaml`.
The vLLM Recipes renderer merges them into the corresponding vLLM recipes.

To validate against a renderer checkout that includes the vLLM SAIL source:

```bash
node scripts/validate-source.mjs --source vllm-sail --dir /path/to/vllm-sail/recipes
```
