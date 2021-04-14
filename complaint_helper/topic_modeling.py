import pandas as pd
import pickle
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords


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

        print(f"Success ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)

    def show_topics(self, vectorizer, lda_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def predict_topic(self, text, vectorizer, lda_model, df_topic_keywords):
        vect_text = vectorizer.transform(text)
        topic_probability_scores = lda_model.transform(vect_text)
        topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
        infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
        return infer_topic

    def apply_predict_topic(self, text, vectorizer, lda_model, df_topic_keywords):
        text = [text]
        topic = self.predict_topic(text, vectorizer, lda_model, df_topic_keywords)
        return topic

    def _get_topic_extractor_data(self):
        """

        :return:
        """
        return self.data[self.topic_extract_col]

    def _tfidf_vectorizer(self,
                          save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()
        text_data = self._get_topic_extractor_data()

        stop_words = set(stopwords.words('english'))
        new_stopwords = ['comcast']
        new_stopwords_list = stop_words.union(new_stopwords)
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                           min_df=2,
                                           max_features=self.n_feature,
                                           stop_words=new_stopwords_list)
        tfidf = tfidf_vectorizer.fit_transform(text_data)

        if save:
            path = os.path.join(self.model_output_path, 'tfidf_vectorizer.pkl')
            pickle.dump(tfidf_vectorizer, open(path, 'wb'))

        print(f"TFIDF Success ({int(time.time() - start_time) % 60})s")
        print("*" * 89)
        return tfidf, tfidf_vectorizer

    def _tf_vectorizer(self,
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
        return tf, tf_vectorizer

    def nmf_model(self,
                  save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()

        tfidf, label = self._tfidf_vectorizer()
        nmf = NMF(n_components=self.n_topics,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5).fit(tfidf)

        topic_keywords = self.show_topics(label, nmf)
        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        Topics = ["Service Quality", "TV/Phone Support", "Equipment Failure", "Connection Issue", "Billing Complaint",
                  "Blocking Other Accounts", "Overcharges", "Bandwidth Issue", "Usage Issue", "Speed Issue"]
        df_topic_keywords["Topics"] = Topics

        if save:
            path = os.path.join(self.model_output_path, 'nmf.pkl')
            pickle.dump(nmf, open(path, 'wb'))

        self.data['Topic'] = self.data[self.topic_extract_col].apply(self.apply_predict_topic,
                                                                     args=(label, nmf, df_topic_keywords))

        print(f"NMF Trained Successfully ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)

    def lda_model(self,
                  save: bool = True):
        """

        :param save:
        :return:
        """
        start_time = time.time()

        tf, label_tf = self._tf_vectorizer()
        lda = LatentDirichletAllocation(n_components=self.n_topics,
                                        max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda_fit = lda.fit(tf)

        topic_keywords = self.show_topics(label_tf, lda)
        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        Topics = ["Service Quality", "TV/Phone Support", "Equipment Failure", "Connection Issue", "Billing Complaint",
                  "Blocking Other Accounts", "Overcharges", "Bandwidth Issue", "Usage Issue", "Speed Issue"]
        df_topic_keywords["Topics"] = Topics

        if save:
            path = os.path.join(self.model_output_path, 'lda.pkl')
            pickle.dump(lda_fit, open(path, 'wb'))

        self.data['Topic'] = self.data[self.topic_extract_col].apply(self.apply_predict_topic,
                                                                     args=(label_tf, lda, df_topic_keywords))

        print(f"LDA Trained Successfully ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)