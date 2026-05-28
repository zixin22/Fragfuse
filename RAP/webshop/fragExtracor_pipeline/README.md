# FragExtractor Pipeline

This pipeline converts WebShop attack rows into FragFuse inputs.

For each input row, it:

1. extracts sensitive fragments from `instruction` with an LLM;
2. builds `masked_query` by replacing fragments with `<>`;
3. builds `carrier_query` by appending split fragment chunks to `host_instruction`.

## Input

`dataset_input.json` should be a JSON array. Each row needs at least:

- `instruction`
- `host_instruction`

Other fields such as `id`, `profile`, and `host_fix_number` are copied through.

## Output

`output.json` keeps the original fields and adds:

- `fragment`
- `masked_query`
- `carrier_query`

## Run

```bash
cd webshop/fragExtracor_pipeline
python3 frag_mask_pipeline.py \
  --input dataset_input.json \
  --output output.json \
  --model gpt-4o
```

Optional:

```bash
python3 frag_mask_pipeline.py --input dataset_input.json --output output.json --model gpt-4o --limit 10 --verbose
```
