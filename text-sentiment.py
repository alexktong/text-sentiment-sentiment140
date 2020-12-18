# Classifies texts into positive and negative sentiment.

import random
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd


class TextSentimentClassifier:

    @staticmethod
    def stem_tokenizer(str_input):
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(str_input)

        stemmer = PorterStemmer()
        words = [stemmer.stem(token) for token in tokens]
        return words

    def __init__(self, sample_fraction=0.1):
        # target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
        # ids: The id of the tweet ( 2087)
        # date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
        # flag: The query (lyx). If there is no query, then this value is NO_QUERY.
        # user: the user that tweeted (robotickilldozr)
        # text: the text of the tweet (Lyx is cool)

        # retrieve 10% of data set randomly for train and test
        df = pd.read_csv('sentiment140.zip', names=['target', 'ids', 'date', 'flag', 'user', 'text'],
                         usecols=['target', 'text'], encoding='iso-8859-1',
                         skiprows=lambda i: i > 0 and random.random() > sample_fraction)

        # remove twitter handles
        df.replace(regex='@[\w]*', value='', inplace=True)

        # split dataset into train and test samples
        x_train, x_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=0)

        # fit text features "X variables" of train data into count-matrix and transform to tf-idf representation
        stem_english_stop_words = [self.stem_tokenizer(stop_word)[0] for stop_word in ENGLISH_STOP_WORDS]
        self.tfv = TfidfVectorizer(max_features=3000, strip_accents='unicode', stop_words=stem_english_stop_words,
                                   tokenizer=self.stem_tokenizer)

        x_train_tfv = self.tfv.fit_transform(x_train)

        # determine optimal classifier parameters using GridSearch; estimator determined here:
        # https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

        param_grid = {'alpha': [0.0001, 0.00001],
                      'max_iter': [100, 500, 1000]}
        self.clf = GridSearchCV(estimator=SGDClassifier(n_jobs=-1), param_grid=param_grid, cv=5, n_jobs=-1, refit=True)
        self.clf.fit(x_train_tfv, y_train)

    def predict(self, iterable_input_text):

        # transform "X variables" of test data into tf-idf based on fitted train data
        x_test_tfv = self.tfv.transform(iterable_input_text)

        # classifies iterable of input text into negative "0" and positive "4"
        return self.clf.predict(x_test_tfv)