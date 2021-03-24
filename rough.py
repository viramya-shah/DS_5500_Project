from complaint_helper.sentiment_classifier import SentimentClassifier

if __name__ == "__main__":
    sentimentClassifier = SentimentClassifier(file_name='sentiment_complaints.csv',
                                              data_path='./data/raw_data',
                                              output_data_path='./data/Sentiment Classifier/corpus',
                                              model_path='./data/Sentiment Classifier/model',
                                              complaint_severity=None)

    # sentimentClassifier = SentimentClassifier(file_name='sentiment_complaints.csv',
    #                                       data_path='/content/drive/MyDrive/Courses/DS 5500 Viz/DS_5500_Project/data/raw_data',
    #                                       output_data_path='/content/drive/MyDrive/Courses/DS 5500 Viz/DS_5500_Project/data/Sentiment Classifier/corpus',
    #                                       model_path='/content/drive/MyDrive/Courses/DS 5500 Viz/DS_5500_Project/data/model',
    #                                       complaint_severity=None)
                                          
    sentimentClassifier.run()
    sentimentClassifier.predict("super good")
