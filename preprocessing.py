"""
Definition of the preprocessing class, containing several function for NLP preprocessing.
The function perform the typical NLP preprocessing pipeline : removing noise, stop words,
punctuation, tokenization, stemmization, lemmization.
They have been built for the 'Fake Job posting' data set of Kaggle.
"""


import nltk
import re

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from autocorrect import Speller


class Preprocessing:

    def initial_text_cleaning(self, df, col_name):
        '''
        Modifies the column col_name of given dataframe,
        by removing and replacing special characters,
        removing special fields such as URL, phone, email,
        removing punctuations (except dash),
        splitting attached words, which are separable by a capital letter'''

        # replace Nan values
        df['text_clean'] = df[col_name].fillna('')

        # replace special patterns
        df.text_clean = df.text_clean.str.replace('\xa0', ' ').str.replace('&amp;', 'and')

        # remove special fields : URL, phone number, email address
        df.text_clean = df.text_clean.apply(lambda x: re.sub(r'http\S+', ' ', str(x)))\
            .apply(lambda x: re.sub(r'#URL_\S+', ' ', x))\
            .apply(lambda x: re.sub(r'#PHONE_\S+', ' ', x))\
            .apply(lambda x: re.sub(r'#EMAIL_\S+', ' ', x))

        # remove punctuation
        df.text_clean = df.text_clean.apply(lambda x: re.sub(r"[,.;:@#?!&$/']+\ *", " ", x))

        # split word which aren't separated by a space (when second word starts with a capital letter)
        df.text_clean = df.text_clean.apply(lambda x: re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', x)))

    def my_tokenizer(self, df, col_name):
        """
        Creates a new column "tokens" in the dataframe, from column 'col_name'.
        Creates a list of lowercase words.
        Removes stopwords.
        Removes string containing no letter."""

        # tokenize
        df['tokens'] = df[col_name].apply(lambda x: nltk.word_tokenize(x))
        
        # lowercase
        df['tokens'] = df.tokens.apply(lambda l: [w.lower() for w in l])

        def real_word_filter():
            # load and remove set of stopwords
            stops = set(stopwords.words('English'))
            df['tokens'] = df.tokens.apply(lambda l: [w for w in l if w not in stops])

            # keep only words with at least two letters, and keep those with dash (like 'fast-growing')
            #pattern = r"[a-z]+"
            pattern = r"^[a-z]+[-]?[a-z]+$"
            df['tokens'] = df.tokens.apply(lambda l: [w for w in l if bool(re.match(pattern, w))])

        # remove stop words dans pattern match
        real_word_filter()

        # spell checker
        spell = Speller()
        df['tokens'] = df.tokens.apply(lambda l: [spell(w) for w in l])

        # remove stop words dans pattern match, for a second time after spell checker
        real_word_filter()

    def stem_lem(self, df, col_name, stem_lem):
        """
        Creates a new column in the dataframe named 'stem_lem', containing the stemmization
        or lemmization of column 'col_name'.
        Parameter stem_lem should be 'stem' in case one wants to have the stemmization
        and "lem" in case one wants to have the lemmization.    
        """
        
        if stem_lem == 'stem':
            porter_stemmer = PorterStemmer()
            df['stem_lem'] = df[col_name].apply(lambda l: [porter_stemmer.stem(w) for w in l])
            
        elif stem_lem == 'lem':
            wordnet_lemmatizer = WordNetLemmatizer()
            df['stem_lem'] = df[col_name].apply(lambda l: [wordnet_lemmatizer.lemmatize(w) for w in l])
            
        else:
            raise ValueError('Cannot recognize parameter "stem_lem". It must be either "stem" or "lem"')

    def preprocess_text(self, df, col_name, stem_lem):
        local_df = df.copy()
        self.initial_text_cleaning(local_df, col_name)
        print('Text cleaning done (1/3).')

        self.my_tokenizer(local_df, 'text_clean')
        print('Tokenization done (2/3).')

        self.stem_lem(local_df, 'tokens', stem_lem)
        print('Stem_Lem done (3/3).')

        return local_df


__author__ = "Justine Weber"
