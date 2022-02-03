import pickle
import random

import gensim
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 500)


def preprocessing(filename):
    df = pd.read_csv(filename)

    # Odrzucanie niepełnych wierszy, ponieważ są one nieznaczną częścią pełnego zbioru.
    df = df.dropna()

    # Wczytanie zmiana typu kolumny na taki użyteczny dla klasyfikatora.
    df['helpful'] = df['helpful'].str.replace("[", "")
    df['helpful'] = df['helpful'].str.replace("]", "")
    df[['helpfulP', 'helpfulN']] = df['helpful'].str.split(',', 1, expand=True)
    df['helpfulP'] = pd.to_numeric(df['helpfulP'])
    df['helpfulN'] = pd.to_numeric(df['helpfulN'])

    # Odrzucenie cechy reviewTime, ponieważ oznacza ona to samo co unixReviewTime a jest mniej praktyczna
    df.drop(["reviewTime", 'helpful'], axis=1, inplace=True)

    # Kodowanie cech tekstowych na liczbowe
    for col in ['reviewerID', 'asin', 'reviewerName']:
        enc = LabelEncoder().fit(df[col])
        df[col] = enc.transform(df[col])

    df.to_csv("reviews_preprocessed_test.csv", index=False)


def process_bow(df: pd.DataFrame):
    vectorizer = CountVectorizer(max_features=150, strip_accents='ascii')

    summary = vectorizer.fit_transform(df['summary']).toarray()
    # print(vectorizer.get_feature_names_out())
    reviewText = vectorizer.fit_transform(df['reviewText']).toarray()
    # print(vectorizer.get_feature_names_out())

    df_bow = pd.DataFrame(np.concatenate((summary, reviewText), axis=1))
    df = pd.concat([df[list(set(df.columns) - {"reviewText", "summary"})], df_bow], axis=1)
    df.columns = df.columns.astype(str)

    return df


def process_w2v(df: pd.DataFrame):
    embeddings = gensim.models.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz",
                                                                 binary=True)
    stopwords = nltk.corpus.stopwords.words('english')

    def get_tokens(text: str):
        tokens = nltk.word_tokenize(text)
        stems = []

        for token in tokens:
            if re.search("\W", token) is not None or token in stopwords:
                continue
            stems.append(token)
        return stems

    def get_doc_vectors(df_data: pd.DataFrame):
        vectors_arr = np.ndarray((df_data.shape[0], embeddings.vector_size), dtype=np.float32)
        for i in df_data.index:
            tokens = get_tokens(df_data[i])
            temp = np.ndarray((len(tokens), embeddings.vector_size), dtype=np.float32)
            for j in range(len(tokens)):
                if tokens[j] in embeddings:
                    temp[j] = embeddings[tokens[j]]

            temp_mean = temp.mean(axis=0)
            temp_mean = np.nan_to_num(temp_mean)
            vectors_arr[i] = temp_mean

        return vectors_arr

    reviewText_vectors = get_doc_vectors(df['reviewText'])
    summary_vectors = get_doc_vectors(df['summary'])
    df_docs_vectors = pd.DataFrame(np.concatenate([reviewText_vectors, summary_vectors], axis=1))

    df = pd.concat([df[list(set(df.columns) - {"reviewText", "summary"})], df_docs_vectors], axis=1)
    df.columns = df.columns.astype(str)
    return df


def scores(y_true, y_pred, name=None, df_compare=pd.DataFrame(np.zeros((5, 3)))):
    measures = ["F1", "Precision", "Recall"]
    df_scores = pd.DataFrame(columns=measures, index=list_of_labels)

    df_scores["F1"] = f1_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels)
    df_scores["Precision"] = precision_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels,
                                             zero_division=0)
    df_scores["Recall"] = recall_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels)

    print(f"=================== Results: {name} ===================")
    print(df_scores.T)


def prepare_data(filename: str, representation_type: str, preprocess: bool):
    if preprocess:
        preprocessing(filename)
        df = pd.read_csv('reviews_preprocessed_test.csv')
    else:
        df = pd.read_csv(filename)

    if representation_type == 'bow':
        df = process_bow(df)
    else:
        df = process_w2v(df)

    y = df["score"]
    X = df[list(set(df.columns) - {"score"})]

    return X, y


if __name__ == "__main__":
    representation = 'w2v' # wybór reprezentacji: w2v / bow
    test_data_path = 'reviews_train.csv' # ścieżka do zbioru testowego

    X_train, y_train = prepare_data('reviews_preprocessed.csv', representation, False)

    if representation == 'bow':
        best_params = {'max_depth': 50, 'max_features': 'auto', 'n_estimators': 350, 'n_jobs': -1, 'random_state': 23}
    else:
        best_params = {'max_depth': 20, 'max_features': 'auto', 'n_estimators': 350, 'n_jobs': -1, 'random_state': 23}

    print('---Training Classifier---')
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)

    X_test, y_test = prepare_data(test_data_path, representation, True)
    X_test = X_test.reindex(columns=X_train.columns.tolist())

    list_of_labels = y_test.unique()
    majority_label = y_test.value_counts().index[0]
    random.seed(23)

    y_pred_random, y_pred_majority = [], []
    for _ in range(len(X_test)):
        y_pred_random.append(random.choice(list_of_labels))
        y_pred_majority.append(majority_label)

    scores(y_test, y_pred_random, name="RandomClassifier")
    scores(y_test, y_pred_majority, name="MajorityClassifier")
    scores(y_test, clf.predict(X_test), name="RandomForestClassifier")
