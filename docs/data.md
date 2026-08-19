# Data contract

## Zenodo deposit

Study data: [10.5281/zenodo.22005545](https://doi.org/10.5281/zenodo.22005545)

The deposit contains processed analysis inputs rather than the original provider downloads. Titles, abstracts, and claim-derived terms have already been condensed into the `document` field, and a corpus-wide BERTopic filter has been applied to the inclusively acquired sector corpora.

## Patent table

Every sector provides `df_id_year_document.csv` with four columns:

| Column | Meaning |
|---|---|
| `id` | Patent publication identifier. |
| `year` | Publication year used as the observation time. |
| `document` | Condensed English-language text used by the sentence encoder. |
| `topic` | Assignment from the earlier corpus-wide filtering model. This column documents preprocessing and is not reused as a moving-window or longer-run topic label. |

The transition pipeline reads `id`, `year`, and `document` and reconstructs all moving-window topics. Values of `-1` in the deposited `topic` column are assignments from the filtering-stage model; they are distinct from second-stage noise created when window-specific topics are linked.

## Company link table

Every sector also provides `df_company_id_year.csv`:

| Column | Meaning |
|---|---|
| `company` | Standardized firm label used in sector acquisition. |
| `id` | Patent publication identifier. |
| `year` | Publication year associated with the firm–patent link. |

The relation is many-to-many because a patent can be linked to multiple firms. These files support firm-level extensions but are not required to calculate RTC or ADTM.

## Audited local contents

The following counts were computed from the local folder deposited on Zenodo on 19 August 2026:

| Sector | Patent rows | Patent years | Patent file | Company-link rows | Companies |
|---|---:|---|---:|---:|---:|
| Automotive | 1,851,341 | 1900–2024 | 1.23 GB | 1,970,137 | 61 |
| Camera | 1,457,933 | 1900–2023 | 0.97 GB | 1,493,430 | 69 |
| Mobile phone | 810,983 | 1901–2024 | 0.23 GB | 873,787 | 46 |
| Robotics | 172,099 | 1910–2024 | 0.11 GB | 172,581 | 37 |
| Semiconductor | 338,008 | 1934–2024 | 0.21 GB | 343,540 | 53 |

These are sector-specific rows, not a count of globally unique patent identifiers: a patent can occur in more than one sector corpus. The paper configurations impose later start years—1930 for camera and automotive, 1960 for mobile phones, 1975 for robotics, and 1965 for semiconductors—when fitting the moving-window models.
