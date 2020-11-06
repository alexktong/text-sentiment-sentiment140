# Classifies texts into positive and negative sentiment.

import random
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import pandas as pd

input_train_data = './sentiment140/sentiment140.zip'


def read_dataset(sample_fraction=0.1):

    # target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    # ids: The id of the tweet ( 2087)
    # date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    # flag: The query (lyx). If there is no query, then this value is NO_QUERY.
    # user: the user that tweeted (robotickilldozr)
    # text: the text of the tweet (Lyx is cool)

    df = pd.read_csv(input_train_data, names=['target', 'ids', 'date', 'flag', 'user', 'text'], usecols=['target', 'text'], encoding='iso-8859-1', skiprows=lambda i: i >0 and random.random() > sample_fraction)

    # remove twitter handles
    df.replace(regex='@[\w]*', value='', inplace=True)

    return df


def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def text_sentiment_classifier(iterable_input_text):

    # 1. extract 1% of main dataset
    df_dataset = read_dataset(sample_fraction=0.01)

    # 2. split dataset into train and test samples
    x_train, x_test, y_train, y_test = train_test_split(df_dataset['text'], df_dataset['target'], test_size=0.2, random_state=0)

    # 3. fit text features "X variables" of train data into count-matrix and transform to tf-idf representation
    stem_ENGLISH_STOP_WORDS = [textblob_tokenizer(stop_word)[0] for stop_word in ENGLISH_STOP_WORDS]
    tfv = TfidfVectorizer(max_features=3000, strip_accents='unicode', stop_words=stem_ENGLISH_STOP_WORDS, tokenizer=textblob_tokenizer)
    x_train_tfv = tfv.fit_transform(x_train)

    # 4. determine optimal classifier parameters using GridSearch
    # estimator determined here: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

    # param_grid = {'max_iter': [1000, 5000]}
    # clf = GridSearchCV(estimator=SGDClassifier(n_jobs=-1), param_grid=param_grid, cv=5, n_jobs=-1, refit=True)
    clf = SGDClassifier(n_jobs=-1)
    clf.fit(x_train_tfv, y_train)

    # 5. transform "X variables" of test data into tf-idf based on fitted train data
    x_test_tfv = tfv.transform(iterable_input_text)

    # 6. classifies iterable of input text into negative "0" and positive "4"
    return clf.predict(x_test_tfv)


# classifies iterable of input text into negative "0" and positive "4"
text_sentiment_classifier(['i don\'t like the food', 'you are doing great', 'i can\'t meet my target'])

# expected output: [0, 4, 0]