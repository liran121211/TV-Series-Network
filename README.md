# 📺 Subtitle-Based TV-Series Rating Prediction  

Predict IMDb ratings for individual **TV-show episodes** by fusing subtitle-level
linguistic signals with structured metadata (cast, runtime, genre, votes, …).
The project demonstrates an end-to-end, fully reproducible pipeline – from raw
subtitle files and IMDb/OMDb dumps to an AutoML-tuned XGBoost regressor.

---

## ✨ Key Features
| Category | Highlights |
|-----------|------------|
| **Data ingestion** | • Bulk IMDb TSVs & OMDb API<br>• English `.srt` / `.ass` subtitles via [subdl.com](https://subdl.com/) API |
| **NLP pipeline** | • spaCy & NLTK tokenisation<br>• Sentiment (TextBlob)<br>• Emotion distribution (NRCLex)<br>• TF-IDF keywords & repetition-rate<br>• POS / tense / pronoun ratios |
| **Metadata engine** | Cast entropy, runtime, episode-index, multi-genre flag, vote count … |
| **AutoML** | [FLAML](https://github.com/microsoft/FLAML) searches XGBoost, LightGBM, Extra-Trees… (300 s budget) |
| **Reproducibility** | Deterministic 5-fold **GroupKFold** (per-series), `requirements.txt`, joblib model artefacts |
| **Explainability** | SHAP plots & fold-wise feature-importance heat-maps |

---

## 🗄️ Repository Layout
```
subtitle-based-tv-prediction/
├─ Data/                    # small sample subset (full data via scripts)
  ├─ (All .srt/.json metadata)
├─ Utils/
  ├─NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
├─ Config.py
├─IMDB_Analyzer.py
├─RatingPredictor.py
├─Subtitles_Analyzer.py
├─Subtitles_Downloader.py
└─ README.md
```

---

## 🚀 Quick-start

### 1 . Clone & install
```bash
git clone https://github.com/liran121211/subtitle-based-tv-prediction.git
cd subtitle-based-tv-prediction
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

### 2 . Grab data  
Create a free **OMDb** key and an **subdl.com** token, then:
```bash
export OMDB_API_KEY=xxxxxxxx
export SUBDL_API_TOKEN=yyyyyyyy
```

### 3 . Train
```bash
python scripts/train_pipeline.py         --config configs/exp_default.yaml         --subtitles data/subtitles         --metadata data/metadata.csv
```
Artefacts (`model.joblib`, `vectorizer.joblib`, CV reports) drop into
`outputs/<timestamp>/`.

### 4 . Predict a fresh episode
```python
from src.models.inference import EpisodePredictor
pred = EpisodePredictor("outputs/2025-06-28/model.joblib")
score = pred("some_episode.srt", meta_row_dict)
print(f"Predicted IMDb rating: {score:.2f}/10")
```

---

## 🔬 Main Results

| Metric (5-fold CV) | Mean ± SD |
|--------------------|-----------|
| RMSE ↓             | **0.582 ± 0.023** |
| MAE  ↓             | 0.410 ± 0.012 |
| R²   ↑             | **0.797 ± 0.018** |

Top global features (SHAP): `votes`, `runtime_minutes`, `cast_size`,
`emotion_disgust`, `pos_PRP`, `topic_2`.

---

## 🛣️ Road-map
* Multilingual support (mT5 / XLM-R fine-tune)  
* Multimodal fusion – shot-length variance, audio energy  
* Streamlit “what-if” dashboard (subtitle edits ↔ rating deltas)

---

## 🤝 Contributing
Pull-requests are welcome! Please open an issue first to discuss proposed
changes. Make sure to run `pre-commit run --all-files` before pushing.

---

## 📜 Citation
If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{Smadja2025Subtitles,
  title     = {Predicting TV Series Popularity through Subtitles Linguistic Features and IMDb Metrics},
  author    = {Liran Smadja},
  booktitle = {Proc.\ AAMAS},
  year      = {2025}
}
```

---

## 📝 License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<p align="center"><sub>© 2025 Liran Smadja • Built with ❤ and caffeine</sub></p>
