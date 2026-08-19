from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import EXTRA_STOP_WORDS, ensure_directory, parse_list, read_patents, stable_union


def _window_configuration(
    start: int,
    end: int,
    nominal_min_cluster_size: int,
    effective_min_cluster_size: int,
) -> dict:
    return {
        "start_year": start,
        "end_year": end,
        "embedding_model": "all-MiniLM-L6-v2",
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        },
        "hdbscan": {
            "nominal_min_cluster_size": nominal_min_cluster_size,
            "effective_min_cluster_size": effective_min_cluster_size,
            "min_samples": effective_min_cluster_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
    }


def _topic_information(topic_model) -> pd.DataFrame:
    topic_info = topic_model.get_topic_info().copy()
    topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()
    columns = ["Representation", "KeyBERT", "MMR", "POS"]
    topic_info["keywords"] = topic_info.apply(
        lambda row: stable_union(parse_list(row[column]) for column in columns), axis=1
    )
    topic_info["keywords"] = topic_info["keywords"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    return topic_info


def aggregate_topic_information(output_dir: Path, start_year: int, end_year: int, window_size: int) -> Path:
    records: list[pd.DataFrame] = []
    for start in range(start_year, end_year - window_size + 2):
        end = start + window_size - 1
        path = output_dir / "windows" / f"{start}_{end}" / "topic_info.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing topic information for window {start}-{end}: {path}")
        frame = pd.read_csv(path)
        frame["year_window"] = f"{start}-{end}"
        frame = frame.rename(columns={"Topic": "topic"})
        records.append(frame[["topic", "keywords", "year_window"]])
    aggregate = pd.concat(records, ignore_index=True)
    target = output_dir / "topic_info_all_year_window.csv"
    aggregate.to_csv(target, index=False)
    return target


def fit_rolling_windows(
    input_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    window_size: int,
    min_cluster_size: int,
    overwrite: bool = False,
    save_models: bool = False,
    save_embeddings: bool = False,
) -> None:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction import text
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if end_year - start_year + 1 < window_size:
        raise ValueError("The observation period must contain at least one complete window")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")

    output_dir = ensure_directory(output_dir)
    windows_dir = ensure_directory(output_dir / "windows")
    patents = read_patents(input_path, start_year, end_year)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    stop_words = list(text.ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS))

    for start in range(start_year, end_year - window_size + 2):
        end = start + window_size - 1
        window_dir = ensure_directory(windows_dir / f"{start}_{end}")
        marker = window_dir / "completed.json"
        if marker.exists() and not overwrite:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved["hdbscan"]["nominal_min_cluster_size"] != min_cluster_size:
                raise ValueError(
                    f"Existing window {start}-{end} uses a different minimum cluster size; "
                    "select a separate output directory"
                )
            print(f"Skipping completed window {start}-{end}")
            continue

        window = patents.loc[patents["year"].between(start, end)].copy().reset_index(drop=True)
        if len(window) < 30:
            raise ValueError(f"Window {start}-{end} has only {len(window)} documents")
        documents = window["document"].tolist()
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        if save_embeddings:
            np.save(window_dir / "embeddings.npy", np.asarray(embeddings))

        cap = max(2, int(len(window) / 15))
        effective_min_cluster_size = min(min_cluster_size, cap)
        vectorizer = CountVectorizer(
            stop_words=stop_words, min_df=2, ngram_range=(1, 2)
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=effective_min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
                core_dist_n_jobs=1,
            ),
            vectorizer_model=vectorizer,
            representation_model={
                "KeyBERT": KeyBERTInspired(),
                "MMR": MaximalMarginalRelevance(diversity=0.3),
                "POS": PartOfSpeech("en_core_web_sm"),
            },
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
            low_memory=True,
        )
        hard_topics, probabilities = topic_model.fit_transform(documents, embeddings)
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2:
            raise ValueError(f"Expected a probability matrix, received shape {probabilities.shape}")
        topic_ids = sorted(int(topic) for topic in topic_model.get_topics() if int(topic) != -1)
        if len(topic_ids) != probabilities.shape[1]:
            raise ValueError(
                "BERTopic probability columns could not be matched to non-noise topic identifiers"
            )

        np.save(window_dir / "topic_probabilities.npy", probabilities)
        (window_dir / "probability_topic_ids.json").write_text(
            json.dumps(topic_ids), encoding="utf-8"
        )
        assignments = window[["id", "year"]].copy()
        assignments["topic"] = hard_topics
        assignments.to_csv(window_dir / "df_id_year_topic.csv", index=False)

        topic_info = _topic_information(topic_model)
        topic_info.to_csv(window_dir / "topic_info.csv", index=False)

        annual_sums = []
        for year in sorted(window["year"].unique()):
            mask = window["year"].to_numpy() == year
            annual_sums.append([int(year), *probabilities[mask].sum(axis=0).tolist()])
        sum_vectors = pd.DataFrame(
            annual_sums, columns=["year", *[f"topic_{topic}" for topic in topic_ids]]
        )
        sum_vectors.to_csv(window_dir / "sum_vectors.csv", index=False)

        if save_models:
            topic_model.save(
                window_dir / "bertopic_model",
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=embedding_model,
            )
        config = _window_configuration(
            start, end, min_cluster_size, effective_min_cluster_size
        )
        marker.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Completed window {start}-{end}: {len(topic_ids)} local topics")

    aggregate_topic_information(output_dir, start_year, end_year, window_size)
    run_config = {
        "input": str(input_path),
        "start_year": start_year,
        "end_year": end_year,
        "window_size": window_size,
        "nominal_min_cluster_size": min_cluster_size,
    }
    (output_dir / "first_stage_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
