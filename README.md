# ETSP-project

## Directory Structure

- **`raw_data`**：Contains the original, unprocessed datasets. Currently includes:
  - GND: `GND-Subjects-tib-core.json` (core subject data)
  - TIBKAT: `train\` and `test\` directory (development-stage data files)

- **`cleaned_data`**：Intended to store data after cleaning/preprocessing.

- **`code`**：Holds scripts for data extraction, cleaning, analysis, etc.


# GND Dataset

```bash
processed_data/GND/
│
├── cleaning/
│   └── cleaned_GND.jsonl
│
├── translating/
│   ├── detect_de.csv
│   ├── detect_de.numbers
│   ├── german_to_english.csv
│   ├── translated_GND.jsonl
│   ├── translated_2_GND.jsonl
│   └── translated_3_GND.jsonl
│
└── embedding/
    ├── subject_ids.json
    ├── subject_texts.json
    └── subject_embeddings.npy
```

```bash
Raw GND JSON → cleaning/cleaned_GND.jsonl
             → translating/detect_de.csv            (auto language detection)
             → translating/german_to_english.csv    (manual/verified mapping)
             → translating/translated_*_GND.jsonl   (progressive translation passes)
             → embedding/subject_texts.json         **final English subject labels**
             → embedding/subject_ids.json           list of subject identifiers
             → embedding/subject_embeddings.npy     SBERT embeddings for subjects
```

- `cleaning/cleaned_GND.jsonl` : Cleaned version of the original GND dataset.
- `translating/detect_de.csv ` : Automatic language detection results for each GND label. To identify which labels still required translation.
- `translating/german_to_english.csv` : German → English mapping table.
- `translating/translated_*_GND.jsonl` : Translated version of the GND subjects. `translated_3_GND.jsonl` is the final result.
- `embedding/subject_texts.json` : Final list of all subject IDs. Defines the subject label vocabulary
- `embedding/subject_ids.json` : Final English subject labels (clean + verified + standardized). English text for each subject (used for embedding + metadata)
- `embedding/subject_embeddings.npy` : Final subject embedding matrix. Ready-to-use subject embedding matrix

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

# TIBKAT Dataset

```bash
processed_data/TIBKAT/
│
├── cleaning/              # JSON-LD → unified JSONL (per type/lang)
│   └── ...                # (intermediate files, mostly for reproducibility)
│
├── translating/
│   ├── train/
│   │   └── ...
│   │
│   ├── test/
│   │   └── ...
│   │
│   ├── train_all.jsonl    # merged train split (all types, de+en, fully translated)
│   └── test_all.jsonl     # merged test split (all types, de+en, fully translated)
│
└── embedding/
    ├── tibkat_train_embeddings.npy
    ├── tibkat_train_ids.json
    ├── tibkat_test_embeddings.npy
    └── tibkat_test_ids.json
```

```bash
Raw TIBKAT JSON-LD
    → cleaning/            (per-type JSONL: Article/Book/Conference/Report/Thesis, de/en)
    → translating/train/*.jsonl, translating/test/*.jsonl
        (German → English translation of content.text + title)
    → train_all.jsonl, test_all.jsonl    (merged + fully English text)
    → tibkat_*_ids.json                  (paper_id list per split)
    → tibkat_*_embeddings.npy            (MiniLM embeddings per split)
```

- `cleaning/` : Cleaned version of the original TIBKAT dataset.
- `translating/` : Translated version of the TIBKAT subjects.
- `embedding/` : Ready-to-use embeddings for the TIBKAT papers (one vector per paper)

## tibkat_*_ids.json
```json
[
  "TIBKAT%3A72999130X",
  "TIBKAT%3A1666713376",
  "TIBKAT%3A168396733X",
  ...
]
```
- A list of paper_id values for each split (train / test).
- The index in this list matches the row index in the corresponding embedding matrix.

## tibkat_*_embeddings.npy
- SBERT embeddings for each paper
- Input text: content.text
