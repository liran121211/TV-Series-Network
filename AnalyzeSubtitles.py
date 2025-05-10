import logging
import os
from pathlib import Path

import pysrt
import spacy
import nltk
import numpy as np
from tqdm import tqdm

import Config
import pandas as pd

from io import StringIO
from dotenv import load_dotenv
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
#  Environment Variables
# --------------------------------------------------------------------------- #
load_dotenv()  # loads from .env by default

SPACY_CURRENT_MODEL = os.getenv("SPACY_CURRENT_MODEL")

# --------------------------------------------------------------------------- #
# Load English NLP model
# --------------------------------------------------------------------------- #
nlp = spacy.load(SPACY_CURRENT_MODEL)
nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")


class SubtitlesAnalyzer:
    def __init__(self, srt_path: str, emotion_file_path: str):
        self.srt_path = srt_path
        self.emotion_file_path = emotion_file_path
        self.subs = pysrt.open(srt_path)
        self.emotion_keywords = self.parse_emolex_file()
        self.lines = [sub.text.strip().replace('\n', ' ') for sub in self.subs if sub.text.strip()]
        self.timestamps = [sub.start.ordinal / 1000.0 for sub in self.subs if sub.text.strip()]

        self.standard_scaler = StandardScaler()

        # --------------------------------------------------------------------------- #
        #  Logging configuration
        # --------------------------------------------------------------------------- #
        _LOG_PATH = Path(__file__).with_suffix(".log")

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            handlers=[
                logging.StreamHandler(),  # print()‑like console output
                logging.FileHandler(_LOG_PATH, encoding="utf‑8")
            ],
        )
        self.logger = logging.getLogger("CinemagoerClient")

    def parse_emolex_file(self) -> pd.DataFrame:
        """
        Parse NRC Emotion Lexicon formatted file into a pivoted DataFrame.

        Returns:
            pd.DataFrame: index=words, columns=emotions, values=0 or 1

        Raises:
            FileNotFoundError: if the emotion file path does not exist
            ValueError: if the file format is invalid
        """
        if not os.path.exists(self.emotion_file_path):
            self.logger.exception(f"Emotion file not found at path: {self.emotion_file_path}")
            return pd.DataFrame(columns=["word", "emotion", "value"])

        try:
            with open(self.emotion_file_path, encoding='utf-8', errors='ignore') as f:
                file_data = f.read()
        except Exception as e:
            self.logger.exception(f"Failed to read emotion file: {e}")
            return pd.DataFrame(columns=["word", "emotion", "value"])

        try:
            df = pd.read_csv(StringIO(file_data), sep='\t', header=None, names=["word", "emotion", "value"])

            # Check expected format (must be 3 columns)
            if df.shape[1] != 3:
                self.logger.exception("Invalid format: expected 3 tab-separated columns per line")
                return pd.DataFrame(columns=["word", "emotion", "value"])

            emotion_df = df.pivot(index='word', columns='emotion', values='value')
            emotion_df = emotion_df.fillna(0).astype(int)

            return emotion_df

        except Exception as e:
            self.logger.exception(f"Failed to parse emotion lexicon content: {e}")
            return pd.DataFrame(columns=["word", "emotion", "value"])


# -------------------Linguistic Features-------------------

    def get_line_count(self):
        """Total number of subtitle lines in the episode"""
        return len(self.lines)

    def get_word_count(self):
        """Total word count in the episode"""
        return sum(len(word_tokenize(line)) for line in self.lines)

    def get_avg_sentence_length(self):
        """Average number of words per sentence"""
        all_sentences = [sent for line in self.lines for sent in sent_tokenize(line)]
        sentence_lengths = [len(word_tokenize(sent)) for sent in all_sentences]
        return np.mean(sentence_lengths) if sentence_lengths else 0.0

    def get_lexical_diversity(self):
        """Ratio of unique words to total words"""
        all_words = [word.lower() for line in self.lines for word in word_tokenize(line)]
        return len(set(all_words)) / len(all_words) if all_words else 0.0

    def get_tfidf_keywords(self, top_n=10):
        """Returns top N keywords in the episode based on TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.lines)
        tfidf_sum = tfidf_matrix.sum(axis=0).A1
        keywords_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_sum))
        sorted_keywords = sorted(keywords_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keywords[:top_n]]

    def get_repetition_rate(self, similarity_threshold=0.95):
        """Returns the percentage of lines that are near-duplicates based on cosine similarity"""
        if len(self.lines) < 2:
            return 0.0
        vectorizer = TfidfVectorizer().fit_transform(self.lines)
        sim_matrix = cosine_similarity(vectorizer)
        # Count near-duplicates above threshold (excluding self-similarity)
        duplicates = sum(
            1
            for i in range(len(sim_matrix))
            for j in range(i + 1, len(sim_matrix))
            if sim_matrix[i][j] > similarity_threshold
        )
        possible_pairs = len(self.lines) * (len(self.lines) - 1) / 2
        return duplicates / possible_pairs if possible_pairs else 0.0

    def get_stopword_ratio(self):
        """Returns the ratio of stopwords to total words"""
        stop_words = set(stopwords.words("english"))
        all_words = [word.lower() for line in self.lines for word in word_tokenize(line)]
        if not all_words:
            return 0.0
        stopword_count = sum(1 for word in all_words if word in stop_words)
        return stopword_count / len(all_words)

    # -------------------Sentiment / Style Features-------------------
    def get_avg_sentiment(self):
        """Average sentiment polarity across all lines"""
        polarities = [TextBlob(line).sentiment.polarity for line in self.lines]
        return np.mean(polarities) if polarities else 0.0

    def get_sentiment_std(self):
        """Standard deviation of sentiment polarity"""
        polarities = [TextBlob(line).sentiment.polarity for line in self.lines]
        return np.std(polarities) if polarities else 0.0

    def get_question_rate(self):
        """Percentage of lines that are questions"""
        questions = [1 for line in self.lines if '?' in line]
        return len(questions) / len(self.lines) if self.lines else 0.0

    def get_emotion_distribution(self):
        counts = Counter()

        # Preprocess: lowercase all words from lines
        all_words = set(
            word.lower()
            for line in self.lines
            for word in word_tokenize(line)
        )

        # Iterate over only words in emotion_keywords that appear in all_words
        filtered = self.emotion_keywords.loc[self.emotion_keywords.index.intersection(all_words)]

        # Sum all emotions across matched words
        summed = filtered.sum(axis=0)

        total = summed.sum()
        return {
            emotion: (summed[emotion] / total) if total > 0 else 0.0
            for emotion in filtered.columns
        }

    def get_exclamation_rate(self):
        if not self.lines:
            return 0.0
        exclamations = sum(1 for line in self.lines if line.strip().endswith('!'))
        return exclamations / len(self.lines)

    def get_intensity_score(self):
        all_words = [word.lower() for line in self.lines for word in word_tokenize(line)]
        if not all_words:
            return 0.0
        intense = sum(1 for word in all_words if word in Config.intensity_words)
        return intense / len(all_words)

# -------------------Clustering / Content Features-------------------

    def get_num_clusters(self, n_clusters=5):
        """Number of clusters formed from TF-IDF vectors of subtitle lines"""
        if len(self.lines) < n_clusters:
            return len(self.lines)
        tfidf = TfidfVectorizer().fit_transform(self.lines)
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(tfidf)
        return len(set(km.labels_))

    def get_dominant_cluster_ratio(self, n_clusters=5):
        """Ratio of lines belonging to the most dominant cluster"""
        if len(self.lines) < n_clusters:
            return 1.0
        tfidf = TfidfVectorizer().fit_transform(self.lines)
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(tfidf)
        labels, counts = np.unique(km.labels_, return_counts=True)
        return np.max(counts) / len(self.lines)

    def get_semantic_variance_spacy(self, model_name=SPACY_CURRENT_MODEL):
        """
        Calculates the average semantic variance across all subtitle lines using spaCy embeddings.

        Each subtitle line is embedded into a high-dimensional vector space using a spaCy language model.
        This function then computes the mean variance across all vector dimensions, representing how
        semantically diverse the lines are throughout the episode.

        A higher variance indicates greater semantic variety—i.e., the content of the subtitles
        spans a wider range of topics, tones, or contexts. A lower variance suggests more uniform
        or repetitive language.

        Parameters:
            model_name (str): Name of spaCy model to load.

        Returns:
            float: Average variance across line embeddings.
        """
        try:
            nlp = spacy.load(model_name)
        except Exception as e:
            print(f"Failed to load spaCy model '{model_name}': {e}")
            return 0.0

        vectors = []
        for line in tqdm(self.lines, desc="Embedding lines with spaCy"):
            doc = nlp(line.strip())
            if doc.has_vector:
                vectors.append(doc.vector)

        if len(vectors) < 2:
            return 0.0

        vectors = np.array(vectors)
        variance = np.var(vectors, axis=0).mean()
        return float(variance)

# -------------------Syntactic/Grammatical Features-------------------
    def get_pos_distribution(self):
        """
        Calculate the distribution of parts of speech (nouns, verbs, adjectives) using spaCy.

        Returns:
            dict: POS categories and their relative frequencies.
        """
        import spacy
        from collections import Counter

        nlp = spacy.load(SPACY_CURRENT_MODEL)
        pos_counts = Counter()
        total_tokens = 0

        for line in self.lines:
            doc = nlp(line)
            total_tokens += len(doc)
            for token in doc:
                if token.pos_ == "NOUN":
                    pos_counts["noun"] += 1
                elif token.pos_ == "VERB":
                    pos_counts["verb"] += 1
                elif token.pos_ == "ADJ":
                    pos_counts["adjective"] += 1

        if total_tokens == 0:
            return {"noun": 0.0, "verb": 0.0, "adjective": 0.0}

        return {k: v / total_tokens for k, v in pos_counts.items()}

    def get_all_linguistic_features(self):
        """Returns all computed features as a dictionary"""
        return {
            "line_count": self.get_line_count(),
            "word_count": self.get_word_count(),
            "avg_sentence_length": self.get_avg_sentence_length(),
            "lexical_diversity": self.get_lexical_diversity(),
            "tfidf_keywords": self.get_tfidf_keywords(),
            'repetition_rate': self.get_repetition_rate(),
            'stopword_ratio': self.get_stopword_ratio(),
            }

    def get_all_sentiment_style_features(self):
        return {
            "avg_sentiment": self.get_avg_sentiment(),
            "sentiment_std": self.get_sentiment_std(),
            "question_rate": self.get_question_rate(),
            'emotion_distribution': self.get_emotion_distribution(),
            'exclamation_rate': self.get_exclamation_rate(),
            'intensity_score': self.get_intensity_score(),
        }

    def get_all_clustering_content_features(self):
        return {
            "num_clusters": self.get_num_clusters(),
            "dominant_cluster_ratio": self.get_dominant_cluster_ratio(),
            "semantic_variance_spacy": self.get_semantic_variance_spacy(),
        }

    def get_all_syntactic_grammatical_features(self):
        return {
            "pos_distribution": self.get_pos_distribution(),
        }


instance = SubtitlesAnalyzer(srt_path='Data/Titles/Breaking_Bed/Subtitles/Breaking.Bad.S01E01.720p.BluRay.X264-REWARD-en.srt',
                             emotion_file_path='NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
print(instance.get_all_syntactic_grammatical_features())
