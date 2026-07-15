# Validation

This repository was validated on real managed GPUs against the live Flash control
plane. Every training run below is an actual paid managed run on this account, not a
mock. Costs are the exact realized charges reported by `flash status` (estimate and
realized matched to the cent fraction for every run).

A two-step run validates mechanics and the environment contract. It is deliberately not
evidence of meaningful task-quality improvement, and it does not exercise live serving:
the only serving behavior observed was the deploy-time capability rejection documented
below.

## Training: validated live on real GPUs

All five environments published, passed server-side dry-run and cost estimate, and
completed a bounded two-optimizer-step managed run.

| Example                      | Environment id                      | Dry-run | Est = realized cost | Training run                | State | Hardware            |
| ---------------------------- | ----------------------------------- | ------- | ------------------- | --------------------------- | ----- | ------------------- |
| JSON extraction SFT          | `clay/json-extraction-sft`          | ok      | $0.00010277         | `flash-1784116173-51f483ac` | done  | Vast RTX 5090       |
| Running total SFT            | `clay/running-total-sft`            | ok      | $0.00010277         | `flash-1784116173-3662dcd2` | done  | RunPod RTX 4090 (1) |
| Structured number guess GRPO | `clay/structured-number-guess-grpo` | ok (2)  | $0.00046796         | `flash-1784117226-309b2cfa` | done  | Vast RTX 5090       |
| Thinking math OPD            | `clay/thinking-math-opd`            | ok (3)  | $0.00142643         | `flash-1784117887-34f2433e` | done  | Vast RTX 5090       |
| Thinking science GRPO        | `clay/thinking-science-grpo`        | ok      | $0.00089108         | `flash-1784116176-e0261955` | done  | RunPod (4)          |

Total realized cost across the five final runs: about $0.002991.

Notes:

1. Running total completed after an automatic Flash infrastructure retry (a first
   attempt reported `cuda not available`; Flash re-scheduled it and it finished).
2. Structured number guess: the first managed run `flash-1784116175-e42f4038` failed at
   `start_episode` because a literal JSON example in the system prompt was passed through
   `str.format`. Fixed by building the prompt without `str.format` over the JSON braces
   and adding an opening-prompt regression test; the environment was republished and the
   replacement dry-run `flash-1784117225-9ac1ad97` and run `flash-1784117226-309b2cfa`
   both succeeded.
3. Thinking math OPD: the first run `flash-1784116176-bebb5ca6` failed because the
   thinking model exhausted a 512-token completion budget before emitting a final answer,
   truncating every rollout (`skip reasons: truncated_rollout=31`, one of two optimizer
   updates reached). This was not a teacher failure (0 teacher errors, full coverage on
   the one valid batch) and not a true OOM (about 17.6 GB free). Fixed config-only:
   `max_context_tokens` 1024 to 2048 and `max_completion_tokens` 512 to 1536. Replacement
   dry-run `flash-1784117886-9a64e841` and run `flash-1784117887-34f2433e` both succeeded.
4. Thinking science completed after automatic Flash infrastructure retries (an early
   attempt hit a transient CUDA device error; Flash re-scheduled it and it finished).

Every dry-run also prints a benign schema-advisory line
(`client-only keys: save_at_steps; server-only keys: checkpoint_landmarks`) and still
completes; it reflects a known client/server schema-label drift, not a rejection.

## Deployment, call, undeploy: blocked live by a production serving rollout gap

These steps could not be exercised live at validation time. Deploying any of the five
adapters fails with:

```
serving_contract_unsupported: serving is missing required capabilities revision_provenance
```

This is an external production-infrastructure timing gap, not a defect in this
repository, and it is confirmed to be enforced server-side, independent of client
version:

- The current Flash clients (`0.2.57`) require the serving `/healthz` capabilities
  `immutable_adapter_revisions`, `alias_compare_and_swap`, and `revision_provenance`.
- Production serving `/healthz` currently advertises `immutable_adapter_revisions`,
  `alias_compare_and_swap`, and `thinking_structured_outputs_deferred_v1`, but not
  `revision_provenance`.
- A deploy attempt with an older pre-gate client that has no client-side capability
  check produced the identical `revision_provenance` failure, proving the control-plane
  deploy worker itself now requires the capability.
- The hundreds of adapters that are already `ready` on the same endpoint were deployed
  before the control plane began requiring the capability, so serving itself is healthy;
  only fresh deploys are gated.
- Every failed deploy reported `previous working alias was preserved`, so no partial or
  broken serving state was left behind, and there was nothing to undeploy.

The deploy, call, and undeploy commands in the READMEs and each `call.py` are correct
and require no repository change. Once production serving is redeployed to advertise
`revision_provenance` (tracked outside this repository), `flash deploy RUN_ID`,
`flash chat`, each `call.py`, and `flash undeploy RUN_ID` run as documented. Redeploying
production serving is intentionally out of scope for an examples repository.

## Validation ladder

1. network-free environment and parsing smokes: pass (`uv run pytest`)
2. linter: pass (`uv run ruff check .`)
3. publish each exact environment source: done (all five under `clay/...`)
4. Flash server-side dry-runs: pass for all five
5. cost estimates: recorded above (estimate matched realized cost)
6. bounded managed runs: all five `done` on real GPUs
7. deploy each final adapter with verification: blocked by the serving rollout gap above
8. task-specific `call.py`: blocked (no live deployment to call)
9. undeploy every run: not applicable (no deployment succeeded; alias preserved)
