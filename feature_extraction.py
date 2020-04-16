"""
Definition of the FeatureExtract class, containing several function for extracting features from dataframe.
Extracted features can be from text, from categorical variables, or directly from binary variables.
Several functions enable getting specific datasets depending on the need.
They have been built for the 'Fake Job posting' data set of Kaggle.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtract:

    CATEGORY_COLS = ['employment_type', 'required_experience', 'required_education']
    BINARY_COLS = ['telecommuting', 'has_company_logo', 'has_questions']
    MIN_OCC = 3

    def __init__(self, dataframe):
        self.df = dataframe
        self.y_df = dataframe.fraudulent

    ''' Getting dataframes '''
    def full_df(self):
        job_param_df = self.__job_param_dataset_builder(self.df)

        tfidf_model, text_features_df = self.__text_features_builder(self.df.stem_lem)
        X = pd.concat([job_param_df, pd.DataFrame(text_features_df, columns=tfidf_model.get_feature_names())], axis=1)

        return X, self.y_df

    def text_df(self):
        tfidf_model, text_features_df = self.__text_features_builder(self.df.stem_lem)
        X_df = pd.DataFrame(text_features_df, columns=tfidf_model.get_feature_names())
        return X_df, self.y_df

    def job_param_df(self):
        X_job = self.__job_param_dataset_builder(self.df)
        return X_job, self.y_df

    ''' Getting features '''
    @staticmethod
    def features_from_df(X_df, y_df):
        return np.array(X_df), np.array(y_df)

    ''' Helpers '''
    @staticmethod
    def __job_param_dataset_builder(df, dummy_cols=CATEGORY_COLS, raw_cols=BINARY_COLS):
        series = []
        for col in dummy_cols:
            series.append(pd.get_dummies(df[col]))

        for col in raw_cols:
            series.append(df[col])

        return pd.concat(series, axis=1)

    @staticmethod
    def __text_features_builder(df_text_col):

        def identity_tokenizer(text):
            return text

        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, min_df=FeatureExtract.MIN_OCC)
        features = tfidf.fit_transform(df_text_col).toarray()
        return tfidf, features

    # others to add : number of words in description, in requirements, in benefits.
    # also : number of spelling mistakes


__author__ = "Justine Weber"
