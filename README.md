# TV Series Rating Predictor
Ever wondered if you could tell whether a new TV show will be a hit — just from its first episode? This project lets you do exactly that. By analyzing the words and emotions in a show’s subtitles and combining them with IMDb metadata, you can get an early sense of how audiences might rate it. Whether you’re a producer looking to spot the next big thing, a researcher curious about what makes a show click, or just someone who loves diving deep into TV, this tool helps you peek into the future of a series before it even hits the screen.

## Google NotebookLM (Predicting TV Series Popularity through Subtitles Linguistic Features and IMDb Metrics)

[Click Here To Play](https://notebooklm.google.com/notebook/034d687a-41bf-40b1-aa64-f70e198a5c6c/audio)
---

```
## Project Structure
├── src/
│ ├── Config.py
│ ├── IMDB_Analyzer.py
│ ├── RatingPredictor.py
│ ├── Subtitles_Analyzer.py
│ └── Subtitles_Downloader.py
├── notebooks/
│ └── colab_notebook.ipynb
├── Data/
│ └── Subtitle Files*.srt
│ ├── TV Shows Metadata Files*.json
│ ├── imdb_features_data.csv
│ └── subtitles_features_data.csv
├── Logs/
│ └── (logs will be saved here automatically)
├── requirements.txt
├── .env.template
└── README.md
```

---

## Quick Installation

1. Make sure you have Python 3.10+ installed.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. change the `.env` file and fill in your API keys (OMDb, SubDL, etc.).
4. Download the SpaCy model:
    ```bash
    python -m spacy download en_core_web_lg
    ```

---

## Usage

- To run a full script, open the notebook at `colab_notebook.ipynb` and execute all the cells.
- The notebook demonstrates:
  - Loading subtitles and extracting linguistic features with `SubtitlesAnalyzer`.
  - Analyzing metadata from IMDb with `CinemagoerClient`.
  - Merging data and building feature tables.
  - Predicting IMDb ratings using `RatingPredictor`.

---

## Example Data

In the `Data/[TV_SHOW_NAME]/` folder you’ll find example files:
- subtitles files (`TV_SHOW_NAME_SXX_EXX.srt`).
- episodes and tv show metadata files (`*.json`).

These files allows you to run the full pipeline quickly without needing to download large datasets.

---

## API Keys

The project requires API keys for external data sources:
- OMDb: to retrieve episode and series metadata.
- TMDb: if you need additional metadata enrichment.
- OpenSubtitles/SubDL: to download subtitles automatically.

Add your keys in the `.env` file using the format provided in `.env.template`.

---

## Logs

All modules generate logs automatically in the `Logs/` folder.  
Log files help you track pipeline execution, warnings, and any errors.

---

## Features

- Subtitle linguistic analysis: sentiment, repetition, stopwords ratio, lexical diversity, tense and POS distributions, emotion detection, topic modeling, coherence scores, and more.
- IMDb metadata integration: cast features, genres, runtimes, episode/season info.
- Automated model selection with FLAML for optimal regression performance.
- Cross-validation with proper grouping by TV show.
- Clear, reproducible results.

---

## Contribution

Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

This project integrates ideas from recent research on text and network analysis of TV shows, inspired by works like:
- Hunter et al. (2016): pilot script text networks for viewership prediction.
- Fronzetti Colladon & Naldi (2019): combining dialogue features and character networks.
- Melody Yu (2023): character network metrics correlated with reviews.

---
