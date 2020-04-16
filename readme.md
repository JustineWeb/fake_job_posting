# Swisscom Assignment : Fake Job posting Prediction

## Instructions
***Dataset:*** [Fake Job posting Prediction](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction )

***Task:*** *Predict which job posting is fake . Use both simple and complex NLP methods to do the job.*

***Challenge:*** *It’s an unbalanced dataset*

## Repository description
- 3 python files contain classes
    - `preprocessing.py` : cleaning data, and typical NLP cleaning
    - `feature_extraction.py` : preparing the dataset
        - TFIDF on text 
        - features for categorical variables
    - `models.py` : models and comparison
        - Logistic Regression
        - Decision Tree
        - Random Forest
- File `Swisscom_Assignment.ipynb` provides an example of usage.