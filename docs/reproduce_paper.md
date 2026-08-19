# Reproducing the study

## Baseline cross-industry analysis

`configs/paper_sectors.json` records the five baseline specifications. After extracting the Zenodo deposit under `data/processed_patent_by_sector`, run:

```bash
python scripts/run_study.py \
  --data-root data/processed_patent_by_sector \
  --output-root outputs \
  --sector all
```

The full run embeds several million documents repeatedly across overlapping windows. Run sectors separately when using a managed GPU service, preserve the output folder between sessions, and verify that every expected window has a `completed.json` marker before linking topics.

## Camera restricted-information horizons

Each horizon must be reconstructed independently from patents available no later than that horizon. Do not reuse topics from the full-history run.

```bash
# 1980 horizon
python scripts/fit_window_topics.py --input data/processed_patent_by_sector/camera/df_id_year_document.csv --output outputs/camera_1980 --start-year 1930 --end-year 1980 --window-size 20 --min-cluster-size 1000
python scripts/link_and_measure.py --windows outputs/camera_1980 --output outputs/camera_1980/final --sector camera --start-year 1930 --end-year 1980 --window-size 20 --min-cluster-size 2

# 1995 horizon: use second-stage min_cluster_size 4
# 2008 horizon: use second-stage min_cluster_size 5
```

The first-stage threshold is automatically capped at one fifteenth of the documents when a truncated window is too small.

## Camera parameter-range analysis

The reported baseline uses a 20-year window, first-stage minimum cluster size 1,000, and second-stage minimum cluster size 10. Reproduce the main sensitivity families by changing one component at a time:

- window lengths: 10, 40, and 80 years;
- first-stage minimum cluster sizes: 300, 1,000, and 3,000;
- second-stage minimum cluster sizes with first stage fixed at 1,000: 5, 10, 15, and 20;
- approximately matched final-topic comparisons: 300/32, 1,000/10, and 3,000/4.

Use a separate output directory for every specification. The scripts refuse to combine window folders whose expected range is incomplete and record every clustering parameter in the final metrics file.

## Validation

After a run, execute:

```bash
python scripts/validate_dataset.py --data-root data/processed_patent_by_sector
python -m pytest
```

For every final output, verify that `annual_normalization_audit.json` reports zero saved noise rows and annual normalized sums equal to one within floating-point tolerance.
