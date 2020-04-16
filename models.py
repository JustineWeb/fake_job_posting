"""
Definition of the Models class, containing several skleanr models for predicting fake jobs.
The implemented models are logistic regression, random forest and decision tree.
A function is made for plotting the ORC curves and comparing the models.
They have been built for the 'Fake Job posting' data set of Kaggle.
"""

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

class Models:

    TEST_RATIO = 0.25
    N_PCA_COMP = 2000
    N_ESTIMATORS_RF = 100

    def __init__(self):
        pass

    def data_set_prep(self, X_df, y_df):
        """
        Train / test set splitting, with stratification.
        PCA dimensionality reduction.
        :param X_df: feature dataframe
        :param y_df: label dataframe
        :return: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, stratify=y_df, test_size=self.TEST_RATIO)
        print("Train/Test split done. Start PCA...")

        pca = PCA(n_components=self.N_PCA_COMP)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("PCA done.")

        return X_train_pca, X_test_pca, y_train, y_test

    def random_forest(self, X_tr, X_te, y_tr, y_te):
        # Create the model with 100 trees
        rf_model = RandomForestClassifier(n_estimators=self.N_ESTIMATORS_RF,
                                          bootstrap=True,
                                          max_features='sqrt')
        print("RandomForest created. Start fitting the model...")

        # Fit on training data
        rf_model.fit(X_tr, y_tr)
        print("RF trained. Start prediction...")

        # Predict labels
        y_pred = rf_model.predict(X_te)
        y_pred_proba = rf_model.predict_proba(X_te)[:, 1]
        # Compute score
        score = roc_auc_score(y_te, y_pred)
        score_proba = roc_auc_score(y_te, y_pred_proba)

        print("RF AUC-ROC score : {}".format(score))
        print("RF AUC-ROC score with probas : {}".format(score_proba))

        return rf_model

    def decision_tree(self, X_tr, X_te, y_tr, y_te):
        # Create the model with 100 trees
        tree_model = DecisionTreeClassifier()
        print("Decision tree created. Start fitting the model...")

        # Fit on training data
        tree_model.fit(X_tr, y_tr)
        print("DT trained. Start prediction...")

        # Predict labels
        y_pred = tree_model.predict(X_te)
        y_pred_proba = tree_model.predict_proba(X_te)[:, 1]
        # Compute score
        score = roc_auc_score(y_te, y_pred)
        score_proba = roc_auc_score(y_te, y_pred_proba)

        print("RF AUC-ROC score : {}".format(score))
        print("RF AUC-ROC score with probas : {}".format(score_proba))

        return tree_model

    def logistic_reg(self, X_tr, X_te, y_tr, y_te):
        # Create the model
        logreg = LogisticRegression(solver='lbfgs')
        print("Logistic Reg created. Start fitting the model...")

        # Fit on training data
        logreg.fit(X_tr, y_tr)
        print("LR trained. Start prediction...")

        # Predict labels
        y_pred = logreg.predict(X_te)
        y_pred_proba = logreg.predict_proba(X_te)[:, 1]
        # Compute score
        score = roc_auc_score(y_te, y_pred)
        score_proba = roc_auc_score(y_te, y_pred_proba)

        print("RF AUC-ROC score : {}".format(score))
        print("RF AUC-ROC score with probas : {}".format(score_proba))

        return logreg

    def plot_ROC_curves(self, models, X_test, y_test):

        ax = plt.gca()
        for mod in models:
            plot_roc_curve(mod, X_test, y_test, ax=ax, alpha=0.8)
        plt.show()


__author__ = "Justine Weber"
