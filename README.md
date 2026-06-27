# Mini Parameter Golf

Mini Parameter Golf is a two-hour workshop version of OpenAI Parameter Golf.
The task is to improve [solution.py](solution.py) for lower `val_bpb`.
The original challenge is designed for rigorous evaluation; Mini PG is designed
for educational iteration with lower runtime and cost. In one saved 20-step
Weco run, steps averaged about 2.3 minutes end to end.

For allowed edits and scoring constraints, read [docs/rules.md](docs/rules.md).

## 1. Set Up

If you do not have `uv`, install it from the
[uv install docs](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/WecoAI/mini-pg.git
cd mini-pg
uv sync
uv run weco credits balance
uv run modal setup
```

If `uv run weco credits balance` reports an auth error, run `uv run weco login`.
For Modal authentication and troubleshooting, see [docs/modal.md](docs/modal.md).

If you are using Claude Code to start the Weco run, this dashboard-bridged
session is required before section 2. Start it and do not start a separate
direct run from the original shell.

```bash
uv run weco start claude --headless \
  -p "Read README.md and docs/rules.md, then run ./scripts/weco.sh to optimize solution.py for lower val_bpb. Do not run uv run weco run directly. Monitor the run and report new bests; don't block."
```

Use the Weco dashboard to watch and steer the bridged session.

## 2. Run with Weco

Run the default Mini PG search:

```bash
uv run weco run \
  --source solution.py \
  --eval-command "./scripts/weco_eval.sh solution.py" \
  --metric val_bpb \
  --goal minimize \
  --steps 20 \
  --eval-timeout 210 \
  --save-logs \
  --additional-instructions instructions/weco_instructions.md \
  --output plain \
  --no-open
```

Weco prints a run link or run ID. Use it to watch progress and collect results.
For monitoring, steering, and reporting details, see the appendix in
[docs/appendix.md](docs/appendix.md). For general Weco usage, see
the [Weco docs](https://docs.weco.ai/), the
[Weco CLI repository](https://github.com/wecoai/weco-cli), and the
[Weco examples](https://github.com/WecoAI/weco-cli/tree/main/examples).

## Leaderboard

Lower `val_bpb` is better.

| Record ID | `val_bpb` | `runtime_s` | `artifact_bytes` | Method | Record |
| --- | ---: | ---: | ---: | --- | --- |
| `20260624-1310-baseline` | `3.983848` | `39.016` | `1,265,674` | Baseline 4-layer byte Transformer, learned positions, default quantization, 80 train steps | [record/20260624-1310-baseline](record/20260624-1310-baseline/README.md) |

To submit a leaderboard PR, add a record under [record/](record/) using the
[record format](record/README.md), then update this table.

## Acknowledgements

Mini Parameter Golf is inspired by OpenAI's official
[Parameter Golf](https://github.com/openai/parameter-golf) challenge. Thanks to
the OpenAI Parameter Golf authors and maintainers, including
[@cocohearts](https://github.com/cocohearts),
[@0hq](https://github.com/0hq), and
[@valerio-oai](https://github.com/valerio-oai).

## Citation

If you use Mini Parameter Golf in a workshop, project, or write-up, please cite
this repository:

```bibtex
@software{wecoai_mini_pg_2026,
  author = {{WecoAI}},
  title = {Mini Parameter Golf},
  year = {2026},
  url = {https://github.com/WecoAI/mini-pg}
}
```
