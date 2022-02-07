import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from gensim.models import Word2Vec
# from nltk.stem.lancaster import LancasterStemmer

import re
import string

stop = stopwords.words('english')
df = pd.read_csv("dataset/train.csv")
df2 = pd.read_csv("dataset/test.csv")
toxic=np.array(df["toxic"])
severe_toxic=np.array(df["severe_toxic"])
obscene=np.array(df["obscene"])
threat=np.array(df["threat"])
insult=np.array(df["insult"])
identity_hate=np.array(df["identity_hate"])

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
x = np.array(df["comment_text"].apply(lambda w:w.split()))

trainn = int(0.8 * 159571)

train_x=x[:trainn]
test_x=x[trainn:]

y_train=np.c_[toxic[:trainn],severe_toxic[:trainn],obscene[:trainn],threat[:trainn],insult[:trainn],identity_hate[:trainn]]
y_test=np.c_[toxic[trainn:],severe_toxic[trainn:],obscene[trainn:],threat[trainn:],insult[trainn:],identity_hate[trainn:]]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000,max_df=0.99,min_df=0.01)
tfidf_vect.fit(x)
X_train =  tfidf_vect.transform(train_x)
X_test = tfidf_vect.transform(test_x)

#**********idhar se aage karna hai, label encoding baaki hai sirf, dataset is prepared for that*******

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.predict(X_test)

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
