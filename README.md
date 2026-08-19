# TechTransDetection

TechTransDetection reconstructs an industry's changing technology space from patent text and derives two annual indicators of system-level technological transition:

- **Rate of Topic Change (RTC):** movement of the annual patent distribution toward later-emerging technological topics.
- **Angular Difference in Topic Momentum (ADTM):** directional consistency between consecutive movements of that distribution.

The repository converts the original exploratory notebooks into resumable scripts with explicit inputs, parameters, intermediate files, and validation checks. The implementation is not tied to the five industries in the accompanying study: any patent corpus with an identifier, publication year, and text document can be analyzed.

## Data

The processed data used in the study are archived on Zenodo at [https://doi.org/10.5281/zenodo.22005545](https://doi.org/10.5281/zenodo.22005545). Download and extract the deposit so that the local layout is:

```text
data/processed_patent_by_sector/
├── automotive/
├── camera/
├── mobile_phone/
├── robotics/
└── semiconductor/
```

Each sector contains `df_id_year_document.csv`, the input to the transition pipeline, and `df_company_id_year.csv`, an auxiliary many-to-many link between firms and patents. See [docs/data.md](docs/data.md) for the schema and audited row counts.

The Zenodo patent tables are already document-constructed and topic-filtered. Therefore, replication from the deposit starts with moving-window topic modeling; the corpus-wide filtering scripts are supplied for researchers applying the method to a new, inclusively acquired patent corpus.

## Installation

Python 3.10 or later is required. A CUDA-capable GPU and a high-memory runtime are strongly recommended for the full sector corpora.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[topic-modeling,api,dev]"
python -m spacy download en_core_web_sm
```

## Reproduce one sector

The following commands reproduce the camera baseline from the processed Zenodo data:

```bash
python scripts/fit_window_topics.py \
  --input data/processed_patent_by_sector/camera/df_id_year_document.csv \
  --output outputs/camera \
  --start-year 1930 --end-year 2023 \
  --window-size 20 --min-cluster-size 1000

python scripts/link_and_measure.py \
  --windows outputs/camera \
  --output outputs/camera/final \
  --sector camera \
  --start-year 1930 --end-year 2023 \
  --window-size 20 --min-cluster-size 10 \
  --events-config configs/benchmark_events.json
```

The first command fits BERTopic independently in overlapping 20-year windows advanced by one year. The second links related window-specific topics with HDBSCAN on precomputed Jaccard distances, excludes second-stage noise before annual normalization, and computes RTC and ADTM from the same retained-topic probability simplex.

To run the paper configuration for one or all sectors, use:

```bash
python scripts/run_study.py \
  --data-root data/processed_patent_by_sector \
  --output-root outputs \
  --sector camera

python scripts/run_study.py \
  --data-root data/processed_patent_by_sector \
  --output-root outputs \
  --sector all
```

Completed moving windows are skipped by default, so interrupted runs can be resumed. Use `--overwrite` only when a window must be re-estimated.

## Apply the preprocessing filter to a new corpus

For a new industry, begin with an inclusive CSV containing `id`, `year`, and `document`. The filtering workflow is:

```bash
python scripts/fit_global_filter.py \
  --input data/my_industry_unfiltered.csv \
  --output outputs/my_industry/filter \
  --start-year 1900 --end-year 2024 \
  --min-cluster-size 300

# Set OPENAI_API_KEY in the environment; never place it in a script.
python scripts/score_topic_relevance.py \
  --topic-info outputs/my_industry/filter/topic_info.csv \
  --output outputs/my_industry/filter/topic_scores.csv \
  --sector "my industry" --model gpt-5-mini

python scripts/filter_patents.py \
  --input data/my_industry_unfiltered.csv \
  --patent-topics outputs/my_industry/filter/patent_topics.csv \
  --topic-scores outputs/my_industry/filter/topic_scores.csv \
  --output data/my_industry_filtered.csv
```

The API evaluates four evidence units separately: the combined representative keyword set receives weight 0.4, and each of three representative documents receives weight 0.2. A preliminary topic is retained only when its total score exceeds 0.5. The exact prompt templates are implemented in `techtransdetection.filtering` and documented in [docs/method.md](docs/method.md).

## Outputs

The final stage writes:

- `clustered_topics_all_year.csv`: links every window-specific topic to a longer-run topic;
- `clustering_quality_metrics.json`: cluster count, noise, persistence, DBCV, and intra-cluster Jaccard distance;
- `cluster_sum_mass.csv`: annual topic mass after second-stage noise removal;
- `annual_topic_proportions.csv`: normalized retained-topic probability vectors;
- `rtc_adtm.csv`: raw and centered five-year RTC and ADTM series;
- `rtc_adtm.png`: indicator plot;
- `rtc_adtm_with_benchmark.png`: optional plot against documented transition-event intensity.

See [docs/reproduce_paper.md](docs/reproduce_paper.md) for baseline, truncation, and sensitivity commands.

## Reproducibility notes

- The sentence encoder is fixed to `all-MiniLM-L6-v2`.
- UMAP uses 15 neighbors, five components, cosine distance, minimum distance 0, and random seed 42.
- First-stage HDBSCAN uses Euclidean distance and excess-of-mass selection; its `min_samples` follows the HDBSCAN default and therefore equals the window-specific minimum cluster size.
- Second-stage HDBSCAN uses precomputed Jaccard distances, `min_samples=2`, and excess-of-mass selection.
- Second-stage noise (`cluster_id=-1`) is removed before each annual vector is normalized. The scripts stop if any retained annual vector does not sum to one.
- Exact cluster labels can differ across numerical libraries or CPU/GPU backends even with a fixed seed. Record the environment and do not combine windows generated by different specifications.

## License

The code is released under the MIT License. The Zenodo data remain governed by the license and provider conditions stated in the Zenodo record.
