import ast
import logging
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, make_scorer
)
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMRegressor
except ImportError as e:
    raise ImportError("LightGBM is required. Install with `pip install lightgbm`.") from e

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------- #
#  Logging configuration
# --------------------------------------------------------------------------- #
_LOG_PATH = Path(__file__).parent / "Logs" / (Path(__file__).stem + ".log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),  # print()‑like console output
        logging.FileHandler(_LOG_PATH, encoding="utf‑8")
    ],
)


class RatingPredictor:
    def __init__(self, imdb_path, sub_path, model_out, random_state=42, n_splits=5, max_card=100):
        self.logger = logging.getLogger("Logs/RatingPredictor")
        self.imdb_path = Path(imdb_path)
        self.sub_path = Path(sub_path)
        self.model_out = Path(model_out)
        self.random_state = random_state
        self.n_splits = n_splits
        self.max_cardinality = max_card
        self.pipe = None

    def _load_data(self):
        self.logger.info("Loading CSV files …")
        imdb_df = pd.read_csv(self.imdb_path)
        sub_df = pd.read_csv(self.sub_path)
        return imdb_df, sub_df

    def _merge_data(self, imdb_df, sub_df):
        self.logger.info("Merging on imdbID …")
        sub_df = sub_df.drop(columns=['votes', 'rating', 'tv_show_name'], errors='ignore')
        sub_df['imdbID'] = sub_df['imdbID'].str.replace('tt', '', regex=False).astype(int)
        return pd.merge(imdb_df, sub_df, on="imdbID", how="inner", suffixes=("_imdb", "_sub"))

    def _clean_target(self, df):
        self.logger.info("Cleaning target …")
        df = df.dropna(subset=["rating"])
        mask = df["rating"].between(1, 10)
        dropped = len(df) - mask.sum()
        if dropped > 0:
            self.logger.warning(f"Dropping {dropped} rows with out-of-range ratings …")
        return df.loc[mask]

    def _prepare_features(self, df):
        y = df["rating"].astype(float)
        leak_cols = {"rating", "rating_imdb", "rating_sub", "imdbID"}
        X = df.drop(columns=[c for c in leak_cols if c in df.columns])

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

        for col in cat_cols[:]:
            if X[col].nunique() > self.max_cardinality:
                self.logger.info(
                    f"Column '{col}' has high cardinality ({X[col].nunique()}); label-encoding instead of OHE.")
                X[col] = X[col].astype("category").cat.codes
                num_cols.append(col)
                cat_cols.remove(col)

        return X, y, cat_cols, num_cols

    def _build_pipeline(self, cat_cols, num_cols):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])

        pre = ColumnTransformer([
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ])

        model = LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.pipe = Pipeline([
            ("pre", pre),
            ("gb", model),
        ])

    def _evaluate(self, X, y, df):
        self.logger.info("Running cross-validation with multiple metrics …")

        if "tv_show_name" in df.columns:
            cv = GroupKFold(n_splits=self.n_splits).split(X, y, df["tv_show_name"])
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, y)

        scorers = {
            "RMSE": make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                                greater_is_better=False),
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "R2": make_scorer(r2_score),
            "ExplainedVar": make_scorer(explained_variance_score),
        }

        results = cross_validate(self.pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1)

        for metric in scorers:
            values = results[f'test_{metric}']
            score = -values if metric in ["RMSE", "MAE"] else values
            self.logger.info(f"{metric}: {score.mean():.4f} ± {score.std():.4f}")

    def _save_model(self):
        self.logger.info(f"Saving model to {self.model_out}")
        self.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, self.model_out)
        self.logger.info("Model saved ✔")

    def predict_rating(self, imdb_id: str, subtitle_features: dict):
        """
        Predicts a rating given an IMDB title ID (e.g. 'tt1234567') and a dictionary of subtitle features.
        """
        try:
            if not self.pipe:
                self.logger.info("Loading model from disk …")
                self.pipe = joblib.load(self.model_out)

            imdb_numeric_id = int(imdb_id.replace("tt", ""))
            imdb_df = pd.read_csv(self.imdb_path)
            row = imdb_df[imdb_df["imdbID"] == imdb_numeric_id].copy()

            if row.empty:
                raise ValueError(f"IMDB ID {imdb_id} not found in IMDB dataset.")

            for k, v in subtitle_features.items():
                row[k] = v

            leak_cols = {"rating", "rating_imdb", "rating_sub", "imdbID"}
            features = row.drop(columns=[c for c in leak_cols if c in row.columns], errors="ignore")
            prediction = self.pipe.predict(features)[0]

            self.logger.info(f"Predicted rating for {imdb_id}: {prediction:.2f}")
            return prediction
        except Exception as e1:
            self.logger.exception(f"Failed to predict rating for {imdb_id}: {e1}")
            return None

    def run(self):
        try:
            imdb_df, sub_df = self._load_data()
            df = self._merge_data(imdb_df, sub_df)
            df = self._clean_target(df)
            df = self.transform_tfidf_keywords_column(local_df=df, column_name='tfidf_keywords', max_vocab_size=500)
            df = self.expand_dict_column(local_df=df, column_name='emotion_distribution', prefix="emotion")
            df = self.expand_dict_column(local_df=df, column_name='pos_distribution', prefix="pos")
            df = self.expand_dict_column(local_df=df, column_name='tense_distribution', prefix="tense")
            df = self.expand_topic_column(local_df=df, column_name='topic_distribution', prefix='topic', expected_length=5)
            X, y, cat_cols, num_cols = self._prepare_features(df)
            self._build_pipeline(cat_cols, num_cols)
            self._evaluate(X, y, df)
            self.logger.info("Fitting on full dataset …")
            self.pipe.fit(X, y)
            self._save_model()
        except Exception as e2:
            self.logger.exception(f"Failed to execute rating predictor pipeline: {e2}")

    def transform_tfidf_keywords_column(self, local_df: pd.DataFrame, column_name: str, max_vocab_size: int = 500) -> pd.DataFrame:
        """
        Transforms a column of stringified TF-IDF keyword lists into binary columns (one per keyword),
        based on a global vocabulary of most frequent keywords.

        Parameters:
        - df: DataFrame containing the keyword column
        - column_name: Name of the column with keyword lists (as string)
        - max_vocab_size: Maximum number of keywords to include as features

        Returns:
        - Modified DataFrame with binary keyword features
        """

        def safe_parse_keywords(s):
            try:
                return ast.literal_eval(s) if isinstance(s, str) else []
            except Exception:
                return []

        # Parse all lists from the column
        parsed_keywords = local_df[column_name].apply(safe_parse_keywords)

        # Build global vocabulary from all rows
        flat_keywords = [kw for sublist in parsed_keywords for kw in sublist]
        vocab_counter = Counter(flat_keywords)
        vocab = [kw for kw, _ in vocab_counter.most_common(max_vocab_size)]

        # Create binary features for each keyword
        def keyword_to_vector(keywords, vocab):
            return {f"kw_{word}": int(word in keywords) for word in vocab}

        self.logger.info(f"Converting {column_name} into  Binary Vector from Global Keyword Vocabulary...")
        keyword_features = parsed_keywords.apply(lambda kws: keyword_to_vector(kws, vocab))

        # Convert to DataFrame and merge with original
        keyword_df = pd.DataFrame(keyword_features.tolist(), index=local_df.index)
        df = local_df.drop(columns=[column_name])
        df = pd.concat([df, keyword_df], axis=1)

        return df

    def expand_dict_column(self, local_df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
        """
        Expands a column of stringified dictionaries into multiple float columns.

        Parameters:
        - local_df: The input DataFrame
        - column_name: The column to expand
        - prefix: Prefix for new columns

        Returns:
        - DataFrame with new columns added and original column removed.
        """
        import ast

        def safe_parse_dict(s):
            try:
                return ast.literal_eval(s) if isinstance(s, str) else {}
            except Exception:
                return {}

        self.logger.info(f"Converting {column_name} into multiple float columns...")
        parsed_dicts = local_df[column_name].apply(safe_parse_dict)
        expanded_df = pd.json_normalize(parsed_dicts)
        expanded_df.columns = [f"{prefix}_{col}" for col in expanded_df.columns]

        df = local_df.drop(columns=[column_name])
        df = pd.concat([df, expanded_df], axis=1)
        return df

    def expand_list_column(self, local_df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
        """
        Expands a column containing list of floats into separate columns.

        Example: [0.1, 0.2, 0.3] → prefix_0, prefix_1, prefix_2
        """
        self.logger.info(f"Converting {column_name} into floats into separate columns...")
        list_df = pd.DataFrame(local_df[column_name].tolist())
        list_df.columns = [f"{prefix}_{i}" for i in list_df.columns]

        df = local_df.drop(columns=[column_name])
        df = pd.concat([df, list_df], axis=1)
        return df

    def expand_topic_column(self, local_df, column_name: str, prefix: str, expected_length: int):
        """
        Converts a column of stringified float lists into float columns (e.g. topic_0, topic_1, ...)
        """

        # Parse strings to lists
        def safe_parse_list(s):
            try:
                result = ast.literal_eval(s)
                if isinstance(result, list) and len(result) == expected_length:
                    return result
            except:
                pass
            return [0.0] * expected_length  # fallback

        local_df[column_name] = local_df[column_name].apply(safe_parse_list)

        # Expand to columns
        self.logger.info(f"Converting {column_name} into floats into separate columns...")
        topic_df = pd.DataFrame(local_df[column_name].tolist(), index=local_df.index)
        topic_df.columns = [f"{prefix}_{i}" for i in topic_df.columns]

        # Replace the original column
        df = local_df.drop(columns=[column_name])
        df = pd.concat([df, topic_df], axis=1)
        return df


if __name__ == "__main__":
    predictor = RatingPredictor(
        imdb_path="Data/imdb_features_data.csv",
        sub_path="Data/subtitles_features_data.csv",
        model_out="Data/rating_predictor.pkl"
    )
    predictor.run()
