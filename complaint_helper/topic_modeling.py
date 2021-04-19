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
                 n_topics: int = 5,
                 n_top_words: int = 5):
        """
        This class is used to extract topics from a given text. It also provides functionality to train
        models like LDA, and NMF.
        :param data_file_path: The data file
        :param n_features: Number of features we want
        :param n_topics: Number of topic we need
        :param n_top_words: Number of top words needded
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

    def show_topics(self, vectorizer, lda_model, n_words=5):
        """
        Shows the topics which are extracted from the text
        :param vectorizer: Vectorizer to vectorize the text
        :param lda_model: The model itself
        :param n_words: Number ow words
        :return:  Topic keywords
        """
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def predict_topic(self, text, vectorizer, lda_model):
        """
        The main function that given a text, woudl extract the topics from it
        :param text: The input text
        :param vectorizer: The vectorize
        :param lda_model: The model
        :return: Extracted topic
        """
        vect_text = vectorizer.transform(text)
        topic_probability_scores = lda_model.transform(vect_text)
        topic_keywords = self.show_topics(vectorizer, lda_model)

        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        Topics = ["Service Issue", "Usage", "Overcharge", "Speed Issue", "Billing Issue"]
        df_topic_keywords["Topics"] = Topics

        topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
        infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
        return infer_topic

    def _get_contact_info(self, topic):
        if topic == 'Service Issue':
            return "<b>John from AssistMe @ (XXX)-XXX-XXXX</b>"
        if topic == 'Usage':
            return "<b>Krystin from AssistMe @ (XXX)-XXX-XXXX</b>"
        if topic == "Overcharge":
            return "<b>Matt from AssistMe @ (XXX)-XXX-XXXX</b>"
        if topic == 'Speed Issue':
            return "<b>Meghan from AssistMe @ (XXX)-XXX-XXXX</b>"
        if topic == 'Billing Issue':
            return "<b>Russel from AssistMe @ (XXX)-XXX-XXXX</b>"

    def apply_predict_topic(self, text, vectorizer, lda_model):
        """
        Similar to predict_topic function, except the fact it works with text batches as well
        :param text: The batch of text
        :param vectorizer: The vectoriser
        :param lda_model: The model itself
        :return: The topics for all the text batch
        """
        text = [text]
        topic = self.predict_topic(text, vectorizer, lda_model)
        return topic, self._get_contact_info(topic)

    def _get_topic_extractor_data(self):
        """
        The column in the main data to work with. This function is the part of training process.
        :return:
        """
        return self.data[self.topic_extract_col]

    def _tfidf_vectorizer(self,
                          save: bool = True):
        """
        Trains a TF-IDF vectorizer.
        :param save: Flag to save the model
        :return: The vectorizer
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
        Trains a TF vectorizer.
        :param save: Flag to save the model
        :return: The vectorizer
        """
        start_time = time.time()
        text_data = self._get_topic_extractor_data()

        stop_words = set(stopwords.words('english'))
        new_stopwords = ['comcast']
        new_stopwords_list = stop_words.union(new_stopwords)
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_feature,
                                        stop_words=new_stopwords_list)
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
        Trains non-matrix factorization model
        :param save: Flag to save the model
        :return: The model
        """
        start_time = time.time()

        tfidf, label = self._tfidf_vectorizer()
        nmf = NMF(n_components=self.n_topics,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5).fit(tfidf)

        if save:
            path = os.path.join(self.model_output_path, 'nmf.pkl')
            pickle.dump(nmf, open(path, 'wb'))

        self.data['Topic'] = self.data[self.topic_extract_col].apply(self.apply_predict_topic, args=(label, nmf))

        print(f"NMF Trained Successfully ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)

    def lda_model(self,
                  save: bool = True):
        """
        Trains Latent Dirichlet Allocation model
        :param save: Flag to save the model
        :return: The model
        """
        start_time = time.time()

        tf, label_tf = self._tf_vectorizer()
        lda = LatentDirichletAllocation(n_components=self.n_topics,
                                        max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda_fit = lda.fit(tf)

        if save:
            path = os.path.join(self.model_output_path, 'lda.pkl')
            pickle.dump(lda_fit, open(path, 'wb'))

        self.data['Topic'] = self.data[self.topic_extract_col].apply(self.apply_predict_topic, args=(label_tf, lda))

        print(f"LDA Trained Successfully ({int(time.time() - start_time) % 60}s)")
        print("*" * 89)
