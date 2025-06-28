# 📺 TV Series Rating Predictor

Welcome to the project for predicting IMDb ratings of TV series using subtitles analysis and IMDb metadata.  
**Goal**: Provide a tool to analyze and estimate the potential success of TV shows from their earliest pilot episodes.

---

## 📦 Project Structure
├── src/
│ ├── Config.py
│ ├── IMDB_Analyzer.py
│ ├── RatingPredictor.py
│ ├── Subtitles_Analyzer.py
│ └── Subtitles_Downloader.py
├── notebooks/
│ └── demo.ipynb
├── Data/
│ └── Subtitle Files
│ ├── TV Shows Metadata Files
│ ├── imdb_features_data.csv
│ └── subtitles_features_data.csv
├── Logs/
│ └── (logs will be saved here automatically)
├── requirements.txt
├── .env.template
└── README.md


---

## 🚀 Quick Installation

1. Make sure you have Python 3.10+ installed.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Copy the `.env.template` file to a new file named `.env` and fill in your API keys (OMDb, SubDL, etc.).
4. Download the SpaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## 📝 Usage

- To run a full demo, open the notebook at `notebooks/demo.ipynb` and execute all the cells.
- The notebook demonstrates:
  - Loading subtitles and extracting linguistic features with `SubtitlesAnalyzer`.
  - Analyzing metadata from IMDb with `CinemagoerClient`.
  - Merging data and building feature tables.
  - Predicting IMDb ratings using `RatingPredictor`.

---

## 🗂️ Example Data

In the `data/example/` folder you’ll find example files:
- A short sample subtitle file (`episode1.srt`).
- Pre-extracted IMDb and subtitle features (`*.csv`).

These examples allow you to run the full pipeline quickly without needing to download large datasets.

---

## 🔑 API Keys

The project requires API keys for external data sources:
- OMDb: to retrieve episode and series metadata.
- TMDb: if you need additional metadata enrichment.
- OpenSubtitles/SubDL: to download subtitles automatically.

Add your keys in the `.env` file using the format provided in `.env.template`.

---

## 📂 Logs

All modules generate logs automatically in the `Logs/` folder.  
Log files help you track pipeline execution, warnings, and any errors.

---

## ✅ Features

- Subtitle linguistic analysis: sentiment, repetition, stopwords ratio, lexical diversity, tense and POS distributions, emotion detection, topic modeling, coherence scores, and more.
- IMDb metadata integration: cast features, genres, runtimes, episode/season info.
- Automated model selection with FLAML for optimal regression performance.
- Cross-validation with proper grouping by TV show.
- Clear, reproducible results.

---

## 🤝 Contribution

Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

This project integrates ideas from recent research on text and network analysis of TV shows, inspired by works like:
- Hunter et al. (2016): pilot script text networks for viewership prediction.
- Fronzetti Colladon & Naldi (2019): combining dialogue features and character networks.
- Melody Yu (2023): character network metrics correlated with reviews.

---

Enjoy exploring and predicting TV show ratings! 🚀
