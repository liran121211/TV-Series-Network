import json
import logging
import os
import re
import zipfile
import pysrt
import pysubs2
import spacy
import nltk
import numpy as np
import Config
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from io import StringIO
from dotenv import load_dotenv
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
#  Utils Variables
# --------------------------------------------------------------------------- #
ZIP_FILE_NOT_FOUND      = -1
UNDEFINED_SUBTITLE_TYPE = -1
SRT_SUBTITLE_TYPE       = 0
ASS_SUBTITLE_TYPE       = 1
# --------------------------------------------------------------------------- #
#  Environment Variables
# --------------------------------------------------------------------------- #
load_dotenv()  # loads from .env by default
SPACY_CURRENT_MODEL = os.getenv("SPACY_CURRENT_MODEL")
EMOTION_FILE_PATH = 'Utils/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

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

        if self.srt_path.endswith('.ass'):
            self.subtitle_type = ASS_SUBTITLE_TYPE
            self.subs = pysubs2.load(srt_path, encoding="utf-8")  # Automatically detects format
        elif self.srt_path.endswith('.srt'):
            self.subtitle_type = SRT_SUBTITLE_TYPE
            self.subs = pysrt.open(srt_path, encoding='iso-8859-1', error_handling=pysrt.ERROR_PASS)
        else:
            self.subtitle_type = UNDEFINED_SUBTITLE_TYPE
            raise TypeError('file type is nor .srt or .ass')

        if len(self.subs) == 0:
            raise AssertionError(f'{srt_path} - file could not be parsed...')

        self.emotion_keywords = self.parse_emolex_file()
        self.lines = [sub.text.strip().replace('\n', ' ') for sub in self.subs if sub.text.strip()]
        self.standard_scaler = StandardScaler()

        if self.subtitle_type == SRT_SUBTITLE_TYPE:
            self.timestamps = [sub.start.ordinal / 1000.0 for sub in self.subs if sub.text.strip()]
        if self.subtitle_type == ASS_SUBTITLE_TYPE:
            self.timestamps = [sub.start / 1000.0 for sub in self.subs if sub.text.strip()]

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
        self.logger = logging.getLogger("Logs/AnalyzeSubtitles")

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

    def get_pronoun_usage(self):
        """
        Calculate the ratio of personal pronouns in the subtitle lines using spaCy.

        Returns:
            float: proportion of pronouns out of all tokens.
        """
        import spacy

        nlp = spacy.load(SPACY_CURRENT_MODEL)
        total_tokens = 0
        pronoun_count = 0

        for line in self.lines:
            doc = nlp(line)
            total_tokens += len(doc)
            pronoun_count += sum(1 for token in doc if token.pos_ == "PRON")

        return pronoun_count / total_tokens if total_tokens > 0 else 0.0

    def get_tense_distribution(self):
        """
        Estimate the distribution of verb tenses: past, present, future using spaCy's morphological analysis.

        Returns:
            dict: Ratio of past, present, and future verbs.
        """
        import spacy
        from collections import Counter

        nlp = spacy.load(SPACY_CURRENT_MODEL)
        tense_counts = Counter()

        for line in self.lines:
            doc = nlp(line)
            for token in doc:
                if token.tag_ in {"VBD", "VBN"}:
                    tense_counts["past"] += 1
                elif token.tag_ in {"VB", "VBP", "VBZ", "VBG"}:
                    tense_counts["present"] += 1
                elif token.tag_ == "MD":
                    tense_counts["future"] += 1

        total = sum(tense_counts.values())
        if total == 0:
            return {"past": 0.0, "present": 0.0, "future": 0.0}

        return {k: v / total for k, v in tense_counts.items()}

# -------------------Structural Features-------------------
    def get_dialogue_density(self):
        """
        Number of subtitle lines per minute of episode.

        Returns:
            float: lines per minute
        """
        duration = self.get_duration_seconds()
        return (len(self.lines) / (duration / 60)) if duration > 0 else 0.0

    def get_duration_seconds(self):
        """
        Total duration of the episode in seconds, inferred from last subtitle timestamp.

        Returns:
            float: duration in seconds
        """
        if not self.subs:
            return 0.0

        if self.subtitle_type == ASS_SUBTITLE_TYPE:
            return self.subs[-1].end / 1000.0  # ordinal is in milliseconds
        if self.subtitle_type == SRT_SUBTITLE_TYPE:
            return self.subs[-1].end.ordinal / 1000.0  # ordinal is in milliseconds:

        # suspensió de pagaments value
        return 0.0

    def get_scene_change_density(self, threshold_seconds=20):
        """
        Estimate number of scene changes per minute based on large time gaps between subtitle entries.

        Parameters:
            threshold_seconds (float): Time gap threshold to consider a scene change.

        Returns:
            float: scene changes per minute
        """
        if len(self.timestamps) < 2:
            return 0.0

        gaps = np.diff(self.timestamps)
        scene_changes = sum(1 for gap in gaps if gap >= threshold_seconds)

        duration_minutes = self.get_duration_seconds() / 60.0
        return scene_changes / duration_minutes if duration_minutes > 0 else 0.0

# -------------------Semantic Features-------------------
    def get_topic_distribution(self, n_topics=5):
        """
        Perform topic modeling using LDA to find distribution of topics in subtitle lines.

        Returns:
            list[float]: topic proportions (length = n_topics)
        """
        if not self.lines:
            return [0.0] * n_topics

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.lines)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        topic_distribution = lda.transform(X).mean(axis=0)
        return topic_distribution.tolist()

    def get_semantic_similarity_score(self, model_name='all-MiniLM-L6-v2'):
        """
        Calculate average semantic similarity between all pairs of subtitle lines.

        Returns:
            float: average pairwise cosine similarity
        """
        if len(self.lines) < 2:
            return 0.0

        model = SentenceTransformer(model_name)
        embeddings = model.encode(self.lines, show_progress_bar=True)

        sim_matrix = cosine_similarity(embeddings)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        return float(np.mean(upper_tri))

    def get_coherence_score(self, model_name='all-MiniLM-L6-v2'):
        """
        Estimate content coherence by computing average similarity between consecutive lines.

        Returns:
            float: average similarity of consecutive subtitle lines
        """
        if len(self.lines) < 2:
            return 0.0

        model = SentenceTransformer(model_name)
        embeddings = model.encode(self.lines, show_progress_bar=True)

        scores = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            scores.append(sim)

        return float(np.mean(scores))

# -------------------------------------------------------------

    def get_all_linguistic_features(self):
        """Returns all computed features as a dictionary"""
        return {
            "line_count": self.get_line_count(),
            "word_count": self.get_word_count(),
            "avg_sentence_length": self.get_avg_sentence_length(),
            "lexical_diversity": self.get_lexical_diversity(),
            "tfidf_keywords": self.get_tfidf_keywords(),
            "repetition_rate": self.get_repetition_rate(),
            "stopword_ratio": self.get_stopword_ratio(),
            }

    def __get_all_sentiment_style_features(self):
        return {
            "avg_sentiment": self.get_avg_sentiment(),
            "sentiment_std": self.get_sentiment_std(),
            "question_rate": self.get_question_rate(),
            "emotion_distribution": self.get_emotion_distribution(),
            "exclamation_rate": self.get_exclamation_rate(),
            "intensity_score": self.get_intensity_score(),
        }

    def __get_all_clustering_content_features(self):
        return {
            "num_clusters": self.get_num_clusters(),
            "dominant_cluster_ratio": self.get_dominant_cluster_ratio(),
            "semantic_variance_spacy": self.get_semantic_variance_spacy(),
        }

    def __get_all_syntactic_grammatical_features(self):
        return {
            "pos_distribution": self.get_pos_distribution(),
            "pronoun_usage": self.get_pronoun_usage(),
            "tense_distribution": self.get_tense_distribution(),
        }

    def __get_all_structural_features(self):
        return {
            "dialogue_density": self.get_dialogue_density(),
            "duration_seconds": self.get_duration_seconds(),
            "scene_change_density": self.get_scene_change_density(),
        }

    def __get_all_semantic_features(self):
        return {
            "topic_distribution": self.get_topic_distribution(),
            "semantic_similarity_score": self.get_semantic_similarity_score(),
            "coherence_score": self.get_coherence_score(),
        }

    def calculate_all_features(self):
        """
        Runs all feature extraction groups with tqdm progress and returns a single flat dictionary.

        Returns:
            dict: feature_name -> value
        """
        results = dict()

        print("Extracting linguistic features...", end='', flush=True)
        results['linguistic_features'] = self.get_all_linguistic_features()
        print("\rExtracting linguistic features...[DONE]", end='\n')

        print("Extracting sentiment style features...", end='', flush=True)
        results['sentiment_style_features'] = self.__get_all_sentiment_style_features()
        print("\rExtracting sentiment style features...[DONE]", end='\n')

        print("Extracting clustering content features...", end='', flush=True)
        results['clustering_content_features'] = self.__get_all_clustering_content_features()
        print("\rExtracting clustering content features...[DONE]", end='\n')

        print("Extracting syntactic grammatical features...", end='', flush=True)
        results['syntactic_grammatical_features'] = self.__get_all_syntactic_grammatical_features()
        print("\rExtracting syntactic grammatical features...[DONE]", end='\n')

        print("Extracting structural features...", end='', flush=True)
        results['structural_features'] = self.__get_all_structural_features()
        print("\rExtracting structural features...[DONE]", end='\n')

        print("Extracting semantic features...", end='', flush=True)
        results['semantic_features'] = self.__get_all_semantic_features()
        print("\rExtracting semantic features....[DONE]", end='')

        return results


def extract_zip_to_same_folder(zip_path, overwrite=False):
    zip_path = os.path.abspath(zip_path)
    output_folder = os.path.dirname(zip_path)
    extracted_files = []

    if os.path.exists(zip_path) is False:
        print(f'❌ zip_path not found - {zip_path}')
        return [ZIP_FILE_NOT_FOUND]

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            extracted_path = os.path.join(output_folder, member.filename)

            if os.path.exists(extracted_path) and not overwrite:
                print(f"Skipping existing file: {extracted_path}")
                continue

            # Create directories if needed
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

            # Extract file
            with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                target.write(source.read())

            extracted_files.append(os.path.abspath(extracted_path))
            print(f"✅ Extracted: {extracted_path}")

    return extracted_files


def extract_features_from_single_subtitle(srt_path: str) -> dict:
    """
    Extract all subtitle-based features from a single subtitle file.

    Args:
        srt_path (str): Path to the .srt or .ass subtitle file.

    Returns:
        dict: Flat dictionary of subtitle features.
    """
    try:
        analyzer = SubtitlesAnalyzer(srt_path=srt_path, emotion_file_path=EMOTION_FILE_PATH)
        features_grouped = analyzer.calculate_all_features()
    except (AssertionError, TypeError) as e:
        print(f"⚠️ Failed to analyze subtitle file: {srt_path}\nError: {e}")
        return {}

    # Flatten the nested dictionary into a single-level dict
    flat_features = {
        f"{feature_name}": feature_value
        for group_dict in features_grouped.values()
        for feature_name, feature_value in group_dict.items()
    }

    return flat_features


def extract_all_subtitles_features(base_path: str, output_csv_path: str, extract_zip_fn) -> pd.DataFrame:
    """
    Extract subtitle-based features for all episodes in a given data directory.

    Args:
        base_path (str): Path to the root data folder.
        output_csv_path (str): Where to save the output CSV.
        extract_zip_fn (Callable): Function to extract srt from zip files.

    Returns:
        pd.DataFrame: A DataFrame containing extracted features for all valid episodes.
    """

    columns = [
        "line_count", "word_count", "avg_sentence_length", "lexical_diversity", "tfidf_keywords",
        "repetition_rate", "stopword_ratio", "avg_sentiment", "sentiment_std", "question_rate",
        "emotion_distribution", "exclamation_rate", "intensity_score", "num_clusters",
        "dominant_cluster_ratio", "semantic_variance_spacy", "pos_distribution", "pronoun_usage",
        "tense_distribution", "dialogue_density", "duration_seconds", "scene_change_density",
        "topic_distribution", "semantic_similarity_score", "coherence_score", "imdbID", "rating", "votes"
    ]

    # Load existing data if exists
    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        existing_ids = set(df_existing['imdbID'].astype(str))
    else:
        df_existing = pd.DataFrame(columns=columns)
        existing_ids = set()

    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) != "Metadata":
            continue

        for file in files:
            if not re.match(r"S\d{1,2}_metadata\.json$", file):
                continue

            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                series_json_data = json.load(f)

            for episode_info in series_json_data.values():
                imdb_id = episode_info.get("imdb_id")
                if not episode_info.get('subtitles_exists'):
                    continue

                if imdb_id in existing_ids:
                    print(f'ℹ️ imdb_id: {imdb_id} already exists, ignoring...')
                    continue

                subtitle_paths = extract_zip_fn(zip_path=episode_info['subtitles_full_path'], overwrite=True)[0]
                if subtitle_paths in (None, "", "ZIP_FILE_NOT_FOUND") or not os.path.exists(subtitle_paths):
                    continue

                print(f"⌛ Extracting  features from {subtitle_paths}")
                try:
                    subtitles_analyzer = SubtitlesAnalyzer(
                        srt_path=subtitle_paths,
                        emotion_file_path=EMOTION_FILE_PATH
                    )
                except AssertionError:
                    continue
                except TypeError:
                    continue

                subtitles_features = subtitles_analyzer.calculate_all_features()
                one_dim_features = {
                    f"{feature_name}": feature_value
                    for feature_group_name, sub_features_dict in subtitles_features.items()
                    for feature_name, feature_value in sub_features_dict.items()
                }

                one_dim_features.update({
                    "imdbID": imdb_id,
                    "rating": episode_info.get("rating", 0),
                    "votes": episode_info.get("votes", 0),
                })

                df_existing = pd.concat([df_existing, pd.DataFrame([one_dim_features])], ignore_index=True)
                df_existing.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # Save after every row

    return df_existing


if __name__ == "__main__":
    extract_all_subtitles_features(
                                base_path=r'C:\Users\mor21\PycharmProjects\BigData_TV_Series_Project\Data',
                                output_csv_path='Data/subtitles_features_data.csv',
                                extract_zip_fn=extract_zip_to_same_folder
                               )

