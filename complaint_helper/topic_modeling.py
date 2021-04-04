import pandas as pd
import pickle
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


class TopicModeling:
    def __init__(self,
                 data_file_path: str = "./data/raw_data/comcast_fcc_complaints_2015.csv",
                 topic_extract_col: str = 'Customer Complaint',
                 model_output_path: str = './output/topic_modeling/',
                 n_features: int = 200,
                 n_topics: int = 10,
                 n_top_words: int = 5):
        """

        :param data_file_path:
        :param n_features:
        :param n_topics:
        :param n_top_words:
        """
        print("Initializing")
        start_time = time.time()
        self.data = pd.read_csv(data_file_path, low_memory=False)
        self.topic_extract_col = topic_extract_col
        self.model_output_path = model_output_path
        self.n_feature = n_features
        self.n_topics = n_topics
        self.top_words = n_top_words

        if not os.path.exists(self.model_output_path):
            os.makedirs(self.model_output_path)

        self.tfidf_vectorizer = None

        print(f"Success ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)

    def _get_topic_extractor_data(self):
        """

        :return:
        """
        return self.data[self.topic_extract_col]

    def tfidf_vectorizer(self,
                         save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()
        text_data = self._get_topic_extractor_data()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                           min_df=2,
                                           max_features=self.n_feature,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(text_data)

        if save:
            path = os.path.join(self.model_output_path, 'tfidf_vectorizer.pkl')
            pickle.dump(tfidf_vectorizer, open(path, 'wb'))

        print(f"TFIDF Success ({int(time.time() - start_time) % 60})s")
        print("*" * 89)
        return tfidf

    def tf_vectorizer(self,
                      save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()
        text_data = self._get_topic_extractor_data()

        text_data = self._get_topic_extractor_data()
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_feature,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(text_data)

        if save:
            path = os.path.join(self.model_output_path, 'tf_vectorizer.pkl')
            pickle.dump(tf_vectorizer, open(path, 'wb'))

        print(f"TF Success ({int(time.time() - start_time) % 60})s")
        print("*" * 89)
        return tf

    def nmf_model(self,
                  save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()

        tfidf = self.tfidf_vectorizer()
        nmf = NMF(n_components=self.n_topics,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5).fit(tfidf)

        if save:
            path = os.path.join(self.model_output_path, 'nmf.pkl')
            pickle.dump(nmf, open(path, 'wb'))

        return tfidf_feature_names =  tfidf_vectorizer.get_feature_names()
        print(f"NMF Trained Successfully ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)

    def lda_model(self):
        pass
