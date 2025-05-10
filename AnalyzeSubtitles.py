import pysrt
import spacy
import nltk
import numpy as np

from textblob import TextBlob
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Load English NLP model
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt_tab')
nltk.download("punkt")

class SubtitlesAnalyzer:
    def __init__(self, srt_path):
        self.srt_path = srt_path
        self.subs = pysrt.open(srt_path)
        self.lines = [sub.text.strip().replace('\n', ' ') for sub in self.subs if sub.text.strip()]
        self.timestamps = [sub.start.ordinal / 1000.0 for sub in self.subs if sub.text.strip()]

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


    def get_all_linguistic_features(self):
        """Returns all computed features as a dictionary"""
        return {
            "line_count": self.get_line_count(),
            "word_count": self.get_word_count(),
            "avg_sentence_length": self.get_avg_sentence_length(),
            "lexical_diversity": self.get_lexical_diversity(),

    def get_all_sentiment_style_features(self):
        return {
            "avg_sentiment": self.get_avg_sentiment(),
            "sentiment_std": self.get_sentiment_std(),
            "question_rate": self.get_question_rate(),
        }

    def get_all_clustering_content_features(self):
        return {
            "num_clusters": self.get_num_clusters(),
            "dominant_cluster_ratio": self.get_dominant_cluster_ratio(),
        }


instance = SubtitlesAnalyzer(srt_path='Data/Titles/Breaking_Bed/Subtitles/Breaking.Bad.S01E01.720p.BluRay.X264-REWARD-en.srt')
print(instance.get_all_features())
