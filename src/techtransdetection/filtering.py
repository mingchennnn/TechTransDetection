from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from .common import EXTRA_STOP_WORDS, ensure_directory, parse_list, read_patents, stable_union


def keyword_prompt(keywords: list[str], sector: str) -> str:
    return (
        "Given the following list of keywords representing a technology topic:\n\n"
        f"{keywords}\n\n"
        f"Decide if this topic is significantly related to {sector} technology.\n\n"
        f"Include 'yes' if it clearly or frequently applies to {sector}, is specialized for {sector}, "
        f"or otherwise strongly supports {sector}-related development or use cases "
        f"(e.g., specialized hardware/software specifically tailored for {sector}, "
        f"advanced solutions critical for {sector} functionality, etc.).\n\n"
        "Say 'no' if the connection is extremely general-purpose or peripheral, "
        f"meaning it lacks direct or strong relevance to {sector} "
        f"(e.g., broad technologies with only potential indirect use in {sector}).\n\n"
        "Just respond with 'yes' or 'no'."
    )


def document_prompt(document: str, sector: str) -> str:
    return (
        "Given the following text describing a technology topic:\n\n"
        f"{document}\n\n"
        f"Decide if this topic is significantly related to {sector} technology.\n\n"
        f"Include 'yes' if it clearly or frequently applies to {sector}, is specialized for {sector}, "
        f"or otherwise strongly supports {sector}-related development or use cases.\n\n"
        "Say 'no' if the connection is extremely general-purpose or peripheral, "
        f"meaning it lacks direct or strong relevance to {sector} "
        "(e.g., a broad or generic solution with only a potential indirect use).\n\n"
        "Just respond with 'yes' or 'no'."
    )


def _binary_answer(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    raise ValueError(f"Expected yes/no response, received: {text!r}")


def fit_global_topics(
    input_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    min_cluster_size: int,
    save_model: bool = False,
) -> None:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction import text
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    output_dir = ensure_directory(output_dir)
    patents = read_patents(input_path, start_year, end_year)
    documents = patents["document"].tolist()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    vectorizer = CountVectorizer(
        stop_words=list(text.ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS)),
        min_df=2,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
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
        calculate_probabilities=False,
        low_memory=True,
    )
    topics, _ = topic_model.fit_transform(documents, embeddings)
    patent_topics = patents[["id", "year"]].copy()
    patent_topics["topic"] = topics
    patent_topics.to_csv(output_dir / "patent_topics.csv", index=False)

    topic_info = topic_model.get_topic_info().copy()
    representation_columns = ["Representation", "KeyBERT", "MMR", "POS"]
    topic_info["keywords"] = topic_info.apply(
        lambda row: stable_union(parse_list(row[column]) for column in representation_columns),
        axis=1,
    )
    topic_info["keywords"] = topic_info["keywords"].apply(json.dumps)
    if "Representative_Docs" in topic_info:
        topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
            lambda value: json.dumps(parse_list(value), ensure_ascii=False)
        )
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)
    if save_model:
        topic_model.save(
            output_dir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=embedding_model,
        )


def score_topic_relevance(
    topic_info_path: Path,
    output_path: Path,
    sector: str,
    model: str = "gpt-5-mini",
    pause_seconds: float = 1.2,
) -> None:
    from openai import OpenAI

    topics = pd.read_csv(topic_info_path)
    required = {"Topic", "keywords", "Representative_Docs"}
    missing = required - set(topics.columns)
    if missing:
        raise ValueError(f"Topic information is missing columns: {sorted(missing)}")

    completed: dict[int, dict] = {}
    if output_path.exists():
        for row in pd.read_csv(output_path).to_dict("records"):
            completed[int(row["Topic"])] = row

    client = OpenAI()

    def classify(prompt: str) -> str:
        error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _binary_answer(response.choices[0].message.content or "")
            except Exception as exc:  # API/network errors are retried, then surfaced.
                error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError("Topic classification failed after three attempts") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for _, topic in topics.iterrows():
        topic_id = int(topic["Topic"])
        if topic_id in completed:
            continue
        keywords = parse_list(topic["keywords"])
        documents = parse_list(topic["Representative_Docs"])
        documents = (documents + ["", "", ""])[:3]
        keyword_result = classify(keyword_prompt(keywords, sector))
        document_results = []
        for document in documents:
            document_results.append(classify(document_prompt(document, sector)))
            if pause_seconds:
                time.sleep(pause_seconds)
        score = (0.4 if keyword_result == "yes" else 0.0) + sum(
            0.2 for result in document_results if result == "yes"
        )
        completed[topic_id] = {
            "Topic": topic_id,
            "keywords_is_sector_related": keyword_result,
            "doc1_sector_related": document_results[0],
            "doc2_sector_related": document_results[1],
            "doc3_sector_related": document_results[2],
            "is_sector_score_total": score,
            "api_model": model,
        }
        pd.DataFrame(completed.values()).sort_values("Topic").to_csv(output_path, index=False)


def filter_patents(
    input_path: Path,
    patent_topics_path: Path,
    topic_scores_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    company_input: Path | None = None,
    company_output: Path | None = None,
) -> None:
    patents = pd.read_csv(input_path)
    assignments = pd.read_csv(patent_topics_path, usecols=["id", "topic"])
    scores = pd.read_csv(topic_scores_path, usecols=["Topic", "is_sector_score_total"])
    valid_topics = set(scores.loc[scores["is_sector_score_total"] > threshold, "Topic"].astype(int))
    patents = patents.drop(columns=["topic"], errors="ignore").merge(
        assignments, on="id", how="inner", validate="one_to_one"
    )
    retained = patents.loc[patents["topic"].astype(int).isin(valid_topics)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retained.to_csv(output_path, index=False)
    if company_input is not None or company_output is not None:
        if company_input is None or company_output is None:
            raise ValueError("company_input and company_output must be provided together")
        links = pd.read_csv(company_input)
        links = links.loc[links["id"].isin(set(retained["id"]))]
        company_output.parent.mkdir(parents=True, exist_ok=True)
        links.to_csv(company_output, index=False)
