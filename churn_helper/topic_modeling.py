import pickle, os
from random import shuffle
import multiprocessing
from multiprocessing import Pool
import csv
import nltk


class IntentUtils:
    def __init__(self,
                 data_path: str = './raw_data/dbpedia',
                 train_file: str = 'train.csv',
                 test_file: str = 'test.csv',
                 class_index_file: str = 'index_to_label.pkl'
                 ):
        self.data_path = data_path
        self.train_file = train_file
        self.test_file = test_file
        self.class_index_file = class_index_file

        self.input_path = os.path.join(self.data_path, 'input')
        self.output_path = os.path.join(self.data_path, 'output')

    def map_classes(self, file_path) -> None:
        """

        :param file_path:
        :return:
        """
        index_to_label = {}
        with open(file_path) as f:
            for i, label in enumerate(f.readlines()):
                index_to_label[str(i + 1)] = label.strip()

        pickle.dump(index_to_label,
                    open(os.path.join(self.output_path,
                                      'index_to_label.pkl'),
                         'wb')
                    )

        return None

    def _transform_instance(self, row, index_to_label):
        """

        :param row:
        :param index_to_label:
        :return:
        """
        cur_row = []
        label = "__label__" + index_to_label[row[0]]
        cur_row.append(label)
        cur_row.extend(nltk.word_tokenize(row[1].lower()))
        cur_row.extend(nltk.word_tokenize(row[2].lower()))
        return cur_row

    def preprocess(self,
                   input_file: str,
                   output_file: str,
                   keep: float = 0.1) -> None:
        """

        :param input_file:
        :param output_file:
        :param keep:
        :return:
        """
        index_to_label = pickle.load(open(os.path.join(self.output_path,
                                                       'index_to_label.pkl'),
                                          'rb')
                                     )
        all_rows = []
        with open(input_file, 'r', encoding="utf-8") as csvinfile:
            csv_reader = csv.reader(csvinfile, delimiter=',')
            for row in csv_reader:
                all_rows.append(row)
        shuffle(all_rows)
        all_rows = all_rows[:int(keep * len(all_rows))]
        transformed_rows = []
        for i in all_rows:
            transformed_rows.append(self._transform_instance(i, index_to_label))

        with open(output_file, 'w', encoding="utf-8") as csvoutfile:
            csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
            csv_writer.writerows(transformed_rows)

    def run(self, file_path='./data/classes.txt'):
        """

        :param file_path:
        :return:
        """

        self.map_classes(file_path=file_path)

        # preprocess the data
        self.preprocess(os.path.join(self.input_path,
                                     self.train_file),
                        os.path.join(self.output_path,
                                     'dbpedia.train'),
                        keep=.2
                        )
        self.preprocess(os.path.join(self.input_path,
                                     self.test_file),
                        os.path.join(self.output_path,
                                     'dbpedia.validation')
                        )
