import pickle
import random

import gensim
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import nltk


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Wczytanie zmiana typu kolumny na taki użyteczny dla klasyfikatora.
    df['helpful'] = df['helpful'].str.replace("[", "")
    df['helpful'] = df['helpful'].str.replace("]", "")
    df[['helpfulP', 'helpfulN']] = df['helpful'].str.split(',', 1, expand=True)
    df['helpfulP'] = pd.to_numeric(df['helpfulP'])
    df['helpfulN'] = pd.to_numeric(df['helpfulN'])

    # Odrzucanie niepełnych wierszy, ponieważ są one nieznaczną częścią pełnego zbioru.
    print("Incomplete rows: ", len(df) - len(df.dropna()))
    df = df.dropna()

    # Odrzucenie cechy reviewTime, ponieważ oznacza ona to samo co unixReviewTime a jest mniej praktyczna
    df.drop(["reviewTime", 'helpful'], axis=1, inplace=True)

    # Kodowanie cech tekstowych na liczbowe
    for col in ['reviewerID', 'asin', 'reviewerName']:
        enc = LabelEncoder().fit(df[col])
        df[col] = enc.transform(df[col])

    return df


def process_bow(df: pd.DataFrame):
    from sklearn.feature_extraction.text import CountVectorizer

    df = preprocessing(df)
    vectorizer = CountVectorizer(max_features=150, strip_accents='ascii')

    summary = vectorizer.fit_transform(df['summary']).toarray()
    # print(vectorizer.get_feature_names_out())
    reviewText = vectorizer.fit_transform(df['reviewText']).toarray()
    # print(vectorizer.get_feature_names_out())

    df_bow = pd.DataFrame(np.concatenate((summary, reviewText), axis=1))
    df_result = pd.concat([df[list(set(df.columns) - {"reviewText", "summary"})], df_bow], axis=1)
    df_result.columns = df_result.columns.astype(str)
    return df_result


def process_w2v(df: pd.DataFrame):
    df = preprocessing(df)
    embeddings = gensim.models.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz",
                                                                 binary=True)
    stopwords = nltk.corpus.stopwords.words('english')

    def get_tokens(text: str):
        tokens = nltk.word_tokenize(text)
        stems = []

        # porter = nltk.PorterStemmer()
        for token in tokens:
            if re.search("\W", token) is not None or token in stopwords:
                continue
            # stems.append(porter.stem(token))
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

    df_result = pd.concat([df[list(set(df.columns) - {"reviewText", "summary"})], df_docs_vectors], axis=1)
    df_result.columns = df_result.columns.astype(str)
    return df_result


def scores(y_true, y_pred, name=None, df_compare=pd.DataFrame(np.zeros((5, 3)))):
    measures = ["F1", "Precision", "Recall"]
    df_scores = pd.DataFrame(columns=measures, index=list_of_labels)

    df_scores["F1"] = f1_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels)
    df_scores["Precision"] = precision_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels,
                                             zero_division=0)
    df_scores["Recall"] = recall_score(y_true, y_pred, average=None, pos_label=None, labels=list_of_labels)

    print(f"=================== Results: {name} ===================")
    # print(pd.DataFrame(df_scores.to_numpy() - df_compare.to_numpy()).T)
    print(df_scores.T)
    # return df_scores


if __name__ == "__main__":
    df = pd.read_csv("reviews_train.csv", nrows=100)
    df = process_bow(df)

    y = df["score"]
    X = df[list(set(df.columns) - {"score"})]

    with open("clf_bow.pkl", "rb") as f:
        clf = pickle.load(f)

    list_of_labels = y.unique()
    majority_label = y.value_counts().index[0]
    random.seed(23)

    y_pred_random, y_pred_majority = [], []
    for _ in range(len(X)):
        y_pred_random.append(random.choice(list_of_labels))
        y_pred_majority.append(majority_label)

    scores(y, y_pred_random, name="RandomClassifier")
    scores(y, y_pred_majority, name="MajorityClassifier")
    scores(y, clf.predict(X), name="Trained Classifier")
