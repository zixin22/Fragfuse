# FragFuse Optimization Notes

This implementation corresponds to the optimization component in the
**Design of FragFuse** section of the paper.

FragFuse has three stages:

1. **Sensitive Fragment Discovery**: identify sensitive fragments `f_t` from a
   prohibited query `q`, replace them with markers `v_t`, and produce the
   masked query `q_mask`.
2. **Carrier Construction & Injection**: place the sensitive fragments inside a
   carrier query `q_car`, whose main body is a benign host query `q_host`.
3. **Fragment Retrieval & Fusion**: issue an attack query `q_att` that contains
   `q_host`, a fusion instruction `I_fuse`, and `q_mask`. The goal is to
   retrieve `q_car` from memory and reconstruct the original prohibited intent.

This directory optimizes the Stage 3 components used by WebShop:

- candidate fusion instructions `I_fuse`;
- marker pairs used to wrap sensitive fragments in `q_car`;
- the interaction between the best fusion instruction and the best carrier
  marker template.

## Optimization Loop

The implementation alternates between two phases.

### Fusion-Instruction Phase

The current best carrier-query template is fixed. The optimizer proposes new
fusion-instruction candidates using LLM rewriting, crossover, and mutation.
Each candidate is evaluated on the surrogate query-fusion task.

### Marker-Template Phase

The current best fusion instruction is fixed. The optimizer proposes new marker
pairs for the carrier-query template. Each marker-template candidate is then
evaluated with the fixed fusion instruction.

### Joint Evaluation

After both phases, the current best fusion instruction and best carrier-query
template are evaluated together. This joint score drives early stopping.

## Scoring

The current code uses the paper-aligned surrogate objective without the
`L_coh` constraint. For each training pair:

- the evaluator builds a carrier-style instruction from `q_host` and the
  sensitive fragment;
- the candidate fusion instruction is used to reconstruct the target text;
- the LLM output is compared against the expected sensitive fragment.

A full fragment match receives `1.0`; partial word overlap receives
`0.8`, `0.5`, `0.3`, or `0.1`. This value is stored as
`fusion_success_score`, and the fusion loss is `L_fus = -fusion_success_score`.
The retrieval loss is `L_ret = 1 - sim(q_att, q_host)`, where `sim` is computed
with `sentence-transformers/all-MiniLM-L6-v2` when available and a lexical
fallback otherwise.

The optimizer still selects higher scores, so it uses
`score = fusion_success_score - lambda_ret * L_ret`, equivalent to minimizing
`lambda_ret * L_ret + L_fus`. `L_coh` is implemented separately but is not
enforced by the current optimizer.

## Output Artifacts

New optimization runs write paper-aligned artifact names and fields:

- `best_fusion_instructions.json` stores optimized fusion-instruction candidates;
- `carrier_query_template_seed.txt` stores the carrier-query marker design seed;
- `fusion_success_score` records the direct fusion-success proxy.
