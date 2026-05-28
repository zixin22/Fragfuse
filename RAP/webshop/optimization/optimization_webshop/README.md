# FragFuse WebShop Optimization

This module optimizes the WebShop FragFuse attack components used in Stage 3:

- `I_fuse`: the fusion instruction that reconstructs the masked query.
- `q_car` marker template: the carrier-query format that stores sensitive fragments.

## Pipeline

1. Load seed fusion instructions and carrier-query templates from `data_webshop/`.
2. Evaluate each candidate on a surrogate WebShop query-fusion task.
3. Alternate between optimizing `I_fuse` and the carrier-query marker template.
4. Save the best candidates and logs under `results/`.

## Scoring

The optimizer maximizes:

```text
score = fusion_success_score - lambda_ret * L_ret
```

where `L_fus = -fusion_success_score`, `L_ret = 1 - sim(q_att, q_host)`, and the default `lambda_ret` is `0.2`. `L_coh` is not enforced in this implementation.

## Run

```bash
cd webshop/optimization/optimization_webshop
python run_optimization.py -g 10 -p 5 --no-plots
```
