
# Text Similarity Detection and Analysis Tool

This repository contains a Python script for detecting and analyzing text similarity between files. It uses text preprocessing, TF-IDF calculations, and Levenshtein distance to identify the degree of similarity between two text files.

## Features

- **Text Preprocessing**:
  - Removes punctuation and stopwords.
  - Tokenizes and stems text for standardization.

- **TF-IDF Computation**:
  - Calculates Term Frequency (TF) and Inverse Document Frequency (IDF) scores.
  - Computes TF-IDF values for words in the text.

- **Levenshtein Distance**:
  - Calculates the similarity between sequences using the Levenshtein algorithm.

- **Similarity Detection**:
  - Detects similarity rates between two text files.
  - Outputs detection rates and preprocessed text for analysis.

## Usage

1. **Preprocess Data**:
   - The `preprocess_data()` function tokenizes and processes text to remove unnecessary elements like stopwords.

2. **Compute Similarity**:
   - Use the `detect()` function to analyze the similarity between two text files:
     ```python
     detect(path_to_files, "file1.txt", "file2.txt")
     ```

3. **Levenshtein Distance**:
   - Use `levenshtein()` to compare sequences and calculate their distance:
     ```python
     levenshtein(seq1, seq2)
     ```

## Files

- `main.py`: The main script for text similarity detection.
- **Input Files**:
  - Place the text files to be analyzed in a specific directory (update paths in the script).

## Dependencies

- `nltk`
- `numpy`

To install dependencies, run:
```bash
pip install nltk numpy
```

## How to Run

1. Place the text files to be compared in the specified directory.
2. Update the paths and filenames in the script as needed.
3. Run the script:
   ```bash
   python main.py
   ```

## Example Output

For the provided files `1.txt` and `2.txt`:
- Detection rate (similarity score).
- Preprocessed content of both files.
