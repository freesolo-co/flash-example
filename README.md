# Flash examples

Five small, end-to-end examples for training and serving LoRA adapters with [Flash](https://github.com/freesolo-co/flash). Each folder keeps the environment, training config, local smoke, deployed call, provenance, and full workflow together.

The organization is inspired by the layered, self-contained approach in [OpenPipe ART examples](https://github.com/OpenPipe/ART/tree/main/examples), including the separation between training and deployment in [tic-tac-toe self play](https://github.com/OpenPipe/ART/tree/main/examples/tic_tac_toe_self_play) and the separation of task utilities from training in [HN title generator](https://github.com/OpenPipe/ART/tree/main/examples/hn_title_generator). No ART code is copied.

## Examples

| Example                                                          | Algorithm |  Turns | Thinking |             Structured output | Difficulty   | Live training |
| ---------------------------------------------------------------- | --------- | -----: | -------: | ----------------------------: | ------------ | ------------- |
| [JSON extraction](examples/json-extraction-sft)                  | SFT       | single |       no | JSON gold plus serving schema | beginner     | done          |
| [Running total](examples/running-total-sft)                      | SFT       |  multi |       no |                  bare integer | beginner     | done          |
| [Structured number guess](examples/structured-number-guess-grpo) | GRPO      |  multi |       no |            strict JSON schema | intermediate | done          |
| [Thinking math](examples/thinking-math-opd)                      | OPD       | single |      yes |            final numeric line | intermediate | done          |
| [Thinking science](examples/thinking-science-grpo)               | GRPO      | single |      yes |           final answer letter | intermediate | done          |

## Live validation status

All five environments were published, dry-run, cost-estimated, and trained end to end on
real managed GPUs (Vast and RunPod) for a total realized cost of about $0.003.
Publication, dry-run, cost estimation, and training are validated live. Deploy was
exercised too, but it currently returns the serving capability rejection documented
below, so deployment itself is not yet validated live.

Deploy, call, and undeploy could not be exercised at validation time: fresh deploys are
currently gated by a production serving rollout that has not yet advertised the
`revision_provenance` capability the control plane now requires. This is a server-side
timing gap, not a repository defect, and it is independent of client version. The deploy,
call, and undeploy commands below are correct and need no change; they run as documented
once serving advertises that capability. See [VALIDATION.md](VALIDATION.md) for the exact
run ids, costs, hardware, two training repairs found and fixed during validation, and the
full deployment-blocker evidence.

## Quickstart

Use Python 3.11 or 3.12.

```bash
uv sync
uv tool install --force "freesolo-flash @ git+https://github.com/freesolo-co/flash.git@c669f0b47aa93801b8ab142b5e05136f6f756aa6"
flash login
flash whoami
```

The examples target the 0.2.57 schema. At validation time the released 0.2.57 wheel still
serialized three retired OPD training fields that the production server rejects for every
algorithm, so live validation used the clean `dev` client pinned above. Switch back to
`uv tool install --force freesolo-flash==0.2.57` once a wheel without those fields is
published.

Run all local checks:

```bash
uv run pytest
uv run ruff check .
```

## Common workflow

The checked-in configs use the published `clay/...` environments validated for this
repository. To modify an environment, publish your copy and replace the checked-in id
with the returned id:

```bash
flash env push --name EXAMPLE_NAME examples/EXAMPLE_NAME
```

Validate and inspect cost before training:

```bash
flash train examples/EXAMPLE_NAME/train.toml --dry-run
flash train examples/EXAMPLE_NAME/train.toml --cost
```

Launch a bounded managed run:

```bash
flash train examples/EXAMPLE_NAME/train.toml --background
flash status RUN_ID --follow
flash log RUN_ID --follow
```

GRPO and OPD runs can expose deployable step checkpoints:

```bash
flash checkpoints RUN_ID
flash deploy RUN_ID/step-N
```

SFT final adapters can be deployed with the run id even when no per-step checkpoint list
appears:

```bash
flash deploy RUN_ID
```

> Deploy, call, and undeploy are currently blocked live by the production serving rollout
> described in [Live validation status](#live-validation-status). The commands here are
> correct and unchanged; run them once serving advertises `revision_provenance`.

The exact deployment source can be `RUN_ID/step-N`, but serving calls use the base
`RUN_ID`:

```bash
flash chat RUN_ID -m "test prompt"
```

Each folder includes a `call.py` for a task-specific deployed check. Set the base URL
printed by `flash deploy` and then run it:

```bash
export FLASH_OPENAI_BASE_URL="https://.../v1"
export FREESOLO_API_KEY="..."
export FLASH_RUN_ID="RUN_ID"
uv run python examples/EXAMPLE_NAME/call.py
```

Tear down serving when finished:

```bash
flash undeploy RUN_ID
```

## Design constraints

- examples target the installed Flash 0.2.57 contract
- every dataset is deterministic and small
- no example needs a user-managed teacher key
- provider and GPU selection remain managed by Flash
- no secret values belong in this repository
- two training steps prove the path, not model quality

See [VALIDATION.md](VALIDATION.md) for the exact live runs and outcomes.
