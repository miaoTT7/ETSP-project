# ETSP-project

## Directory Structure

- **`raw_data`**：Contains the original, unprocessed datasets. Currently includes:
  - GND: `GND-Subjects-tib-core.json` (core subject data)
  - TIBKAT: `train\` and `dev\` directory (development-stage data files)

- **`cleaned_data`**：Intended to store data after cleaning/preprocessing.
  - subject_embeddings.npy：the final file(ready to subject index)

- **`code`**：Holds scripts for data extraction, cleaning, analysis, etc.
