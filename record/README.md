# Submission Records

This directory stores leaderboard records for Mini Parameter Golf submissions.
It is only for submitted solutions and verified result metadata; the benchmark
harness, scripts, docs, and root `solution.py` stay unchanged.

## PR Format

Add one directory per entry:

```text
record/<YYYYMMDD-HHMM-slug>/
  README.md
  solution.py
```

Use UTC time for the `YYYYMMDD-HHMM` prefix.

Use [template.md](template.md) for the entry README.

Before a record is accepted, run:

```bash
./scripts/verify.sh record/<YYYYMMDD-HHMM-slug>/solution.py
```

Then update the leaderboard table in the root [README.md](../README.md) with
the verified `val_bpb`, `runtime_s`, `artifact_bytes`, method summary, and link
to the record.

Do not use metrics copied from an agent summary, Weco dashboard text, or chat
message as final leaderboard evidence. Only a fresh verifier run counts.
