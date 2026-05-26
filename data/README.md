# `data/`

`data/cache/` holds per-day option chain snapshots pulled by
`volengine.backtesting.data_loader`. One parquet per `(symbol, date)`:

```
data/cache/SPY_2024-01-02.parquet
data/cache/SPY_2024-01-03.parquet
...
```

`data/spx_options_snapshot.csv` is the canonical single-day SPX snapshot used
by notebooks 01-09 — once you have a clean trading day cached you can copy or
symlink it here.

The cache is in `.gitignore` — large parquet binaries don't belong in version
control. If you want reproducible runs across collaborators, host the
snapshots in cloud storage (S3, GCS) and fetch on first run.
