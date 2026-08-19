# Computational method

## Optional corpus-wide relevance filter

The study begins from deliberately inclusive firm and keyword searches. For each sector, all condensed documents in the unfiltered corpus are embedded once with `all-MiniLM-L6-v2`, and one BERTopic model is fitted across the complete observation period. This model is used only for relevance filtering; it is separate from the moving-window models used by RTC and ADTM.

UMAP reduces the embeddings to five dimensions using 15 neighbors, cosine distance, minimum distance 0, and random seed 42. HDBSCAN then identifies fine-grained preliminary topics using Euclidean distance and the excess-of-mass rule. Each topic is described by the union of c-TF-IDF, KeyBERT, Maximal Marginal Relevance, and part-of-speech terms and by three BERTopic representative documents.

The OpenAI API receives four independent binary classification requests per topic. The keyword-set prompt is:

> Given the following list of keywords representing a technology topic:
> [KEYWORDS]
> Decide if this topic is significantly related to [SECTOR] technology.
> Include “yes” if it clearly or frequently applies to [SECTOR], is specialized for [SECTOR], or otherwise strongly supports [SECTOR]-related development or use cases (e.g., specialized hardware/software specifically tailored for [SECTOR], advanced solutions critical for [SECTOR] functionality, etc.).
> Say “no” if the connection is extremely general-purpose or peripheral, meaning it lacks direct or strong relevance to [SECTOR] (e.g., broad technologies with only potential indirect use in [SECTOR]).
> Just respond with “yes” or “no”.

The representative-document prompt substitutes one document for the keyword list and omits the parenthetical examples in the positive criterion. A positive keyword judgment contributes 0.4; each positive document judgment contributes 0.2. All patents assigned to topics scoring above 0.5 are retained.

Replication from the Zenodo deposit starts after this step. Re-running a hosted language model is unnecessary for reproducing the reported transition analysis and may not yield byte-identical filtering decisions as model services evolve.

## Moving-window topics

The filtered patent corpus is divided into overlapping 20-year windows advanced one year at a time. Within every window, the documents are embedded with the same sentence model and clustered with BERTopic. The nominal first-stage minimum cluster size is capped at one fifteenth of the available documents in small windows. BERTopic returns a soft membership vector over local topics for every patent, including patents whose hard HDBSCAN assignment is noise.

For every window, the software saves the topic descriptions, patent-to-topic assignments, topic-probability matrix, and annual sums of those probabilities. These intermediate files make the costly first stage resumable and allow the linking and indicator stages to be rerun without recomputing embeddings.

## Linking topics across windows

Each window-specific topic is represented by its combined keyword set. Pairwise Jaccard distances between these sets form the input to a second HDBSCAN model using a precomputed metric, `min_samples=2`, and excess-of-mass selection. Each retained cluster links local configurations across windows into one longer-run technological topic.

The soft membership weights of patents published in each year are aggregated from every window containing that year and then mapped to the longer-run topics. Second-stage noise is removed from this annual mass before normalization. The resulting annual vectors are explicitly validated to sum to one.

## RTC and ADTM

Longer-run topics are ordered by their probability-weighted mean years. For every year, RTC takes the first difference of the weighted median topic rank. ADTM is the angle between the incoming and outgoing changes in the complete annual topic-proportion vector. A zero angle is assigned when either movement vector has zero norm. Both raw indicators are smoothed with a centered, flexible five-year moving average, using shorter windows at the temporal boundaries.
