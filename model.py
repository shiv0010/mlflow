import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings
import sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
import mlflow
import mlflow.sklearn

import cloudpickle
from sys import version_info
import sklearn
import mlflow.pyfunc

DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
dataset = dataset[['sentiment','text']]
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

contractions = pd.read_csv('contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_apply(tweet):

    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet

dataset['processed_text'] = dataset.text.apply(preprocess_apply)
dataset=dataset[:10000]
#print(dataset.head())
'''count=0
for row in dataset.itertuples():
    print("Text:", row[2])
    print("Processed:", row[3])
    count+=1
    if count>10:
        break'''

X_data, y_data = np.array(dataset['processed_text']), np.array(dataset['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size = 0.05, random_state = 0)
print('Data Split done.')
Embedding_dimensions = 100
# Creating Word2Vec training dataset.
Word2vec_train_data = list(map(lambda x: x.split(), X_train))
print('word2vec done.')
word2vec_model = Word2Vec(Word2vec_train_data,vector_size=Embedding_dimensions,workers=8,min_count=5)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))
input_length = 60
vocab_length = 5000

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)
print(type(X_test))
print("X_test.shape :", X_test.shape)


PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# Train and save an SKLearn model
sklearn_model_path = "Sentiment-BiLSTM"

artifacts = {
    "sklearn_model": sklearn_model_path
}

# create wrapper
class SKLearnWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import pickle
        self.sklearn_model = tf.keras.models.load_model(context.artifacts["sklearn_model"])
    
    def predict(self, model, data):
        return self.sklearn_model.predict(data)

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'scikit-learn=={}'.format(sklearn.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'sklearn_env'
}

mlflow_pyfunc_model_path = "sklearn_mlflow_pyfunc_custom1"
mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_path, python_model=SKLearnWrapper(), conda_env=conda_env, artifacts=artifacts)

loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
print(1)
print(2)
# Evaluate the model
import pandas as pd
print(3)
test_predictions = loaded_model.predict(X_test)
print(test_predictions)



mlflow.start_run()

mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, 
                        loader_module=None, 
                        data_path=None, 
                        code_path=None,
                        python_model=SKLearnWrapper(),
                        registered_model_name="Custom_mlflow_model", 
                        conda_env=conda_env,
                        artifacts=artifacts)
mlflow.end_run()