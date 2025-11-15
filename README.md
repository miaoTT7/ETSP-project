# ETSP-project

## Directory Structure

- **`raw_data`**：Contains the original, unprocessed datasets. Currently includes:
  - GND: `GND-Subjects-tib-core.json` (core subject data)
  - TIBKAT: `train\` and `test\` directory (development-stage data files)

- **`cleaned_data`**：Intended to store data after cleaning/preprocessing.
  - subject_embeddings.npy：the final file(ready to subject index)

- **`code`**：Holds scripts for data extraction, cleaning, analysis, etc.


# GND Dataset

```bash
cleaned_data/GND/
│
├── cleaned_GND.jsonl
├── detect_de.csv
├── german_to_english.csv
│
├── translated_GND.jsonl
├── translated_2_GND.jsonl
├── translated_3_GND.jsonl
│
├── subject_ids.json
├── subject_texts.json
└── subject_embeddings.npy
```

```bash
Raw GND → cleaned_GND.jsonl
        → translated_GND*.jsonl    (translation passes)
        → detect_de.csv            (language detection)
        → german_to_english.csv    (verified mapping)
        → subject_texts.json       **(final English labels)**
        → subject_ids.json         (subject vocabulary)
        → subject_embeddings.npy   (SBERT embeddings)
```

- `cleaned_GND.jsonl` : Cleaned version of the original GND dataset.
- `detect_de.csv` : Automatic language detection results for each GND label. To identify which labels still required translation.
- `german_to_english.csv` : German → English mapping table.
- `translated_GND.jsonl`, `translated_2_GND.jsonl`, `translated_3_GND.jsonl` : translated version of the GND subjects. translated_3_GND.jsonl is the final result.
- `subject_ids.json` : Final list of all subject IDs. Defines the subject label vocabulary
- `subject_texts.json` : Final English subject labels (clean + verified + standardized). English text for each subject (used for embedding + metadata)
- `subject_embeddings.npy` : Final subject embedding matrix. Ready-to-use subject embedding matrix

## subject_ids.json
```json
[
  "gnd:4071095-6",
  "gnd:4068097-6",
  "gnd:4011455-7",
  ...
]
```
- A list of all GND subject IDs used in the project.

## subject_texts.json
```json
{
  "gnd:4071095-6": "Financial market",
  "gnd:4068097-6": "Future",
  "gnd:4011455-7": "Architecture",
  ...
}
```
- A dictionary that maps each subject ID to its final English label
- These are the text labels the model uses to build embeddings

## subject_embeddings.npy
- A NumPy matrix containing the SBERT (MiniLM-L6-v2) embeddings for every subject in `subject_ids.json`.
- Vector representation for each subject
