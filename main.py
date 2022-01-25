import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from nltk.corpus import stopwords
# from nltk.stem.lancaster import LancasterStemmer

import re
import string

stop = stopwords.words('english')
df = pd.read_csv("dataset/train.csv")
# stemmer = LancasterStemmer()

#remove alphanumeric
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

# '[%s]' % re.escape(string.punctuation),' ' - replace punctuation with white space
# .lower() - convert all strings to lowercase 
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

# Remove all '\n' in the string and replace it with a space
remove_n = lambda x: re.sub("\n", " ", x)

# Remove all non-ascii characters 
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

# Apply all the lambda functions wrote previously through .map on the comments column
df["comment_text"] = df["comment_text"].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

#stopwords
df["comment_text"] = df["comment_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#base form
# df["comment_text"] = df["comment_text"].apply(lambda x: [stemmer.stem(y) for y in x])
print(df["comment_text"])

#**********idhar se aage karna hai, label encoding baaki hai sirf, dataset is prepared for that*******

# df_test = pd.read_csv("dataset/test.csv")
# X_train = np.array(df["comment_text"])
# # print(df_test.head())
# #print(x_train.shape)
# y_train = np.array(df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])
# X_test = np.array(df_test["comment_text"])
# # print(x_test)
# # print(y_train)
# l_reg = linear_model.LinearRegression()
# model = l_reg.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print("predictions: ", predictions) 
