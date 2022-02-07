import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


##############################################################################


test = pd.read_csv(r'C:\Users\Yash\Desktop\test.csv')
train = pd.read_csv(r"C:\Users\Yash\Desktop\train.csv")

test.info()
train.info()

test.isnull().sum()
train.isnull().sum()

print(train.corr())
print(sns.heatmap(train.corr()))

train.skew()


    
# showing different bar graphs:-->
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i in col:
    print(i)
    print(train[i].value_counts())
    sns.countplot(train[i])
    plt.show()

##############################################################################

# Replace email addresses with 'email'
train['comment_text'] = train['comment_text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

# Replace URLs with 'webaddress'
train['comment_text'] = train['comment_text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
train['comment_text'] = train['comment_text'].str.replace(r'£|\$', 'dollers')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
train['comment_text'] = train['comment_text'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')
   
# Replace numbers with 'number'
train['comment_text'] = train['comment_text'].str.replace(r'\d+(\.\d+)?', 'number')
# Remove punctuation
train['comment_text'] = train['comment_text'].str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
train['comment_text'] = train['comment_text'].str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
train['comment_text'] = train['comment_text'].str.replace(r'^\s+|\s+?$', '')
test['comment_text'] = test['comment_text'].str.replace(r'^\s+/\s+?$' , '')





'''PIE-CHART'''

df_distribution = train[col].sum()\
                          .to_frame()\
                          .rename(columns={0: 'count'})\
                          .sort_values('count')
df_distribution.plot.pie(y = 'count', 
                         title = 'Label distribution over comments', 
                         figsize=(5,5))\
.legend(loc = 'center left', bbox_to_anchor = (1.3, 0.5))


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

for i in range(len(test['comment_text'])):
    test['comment_text'][i] = test['comment_text'][i].lower()
    j = []
    for word in test['comment_text'][i].split():
        j.append(lemmatizer.lemmatize(word, pos="v"))
        test['comment_text'][i] = "".join(j)
        
for i in range(len(train['comment_text'])):
    train['comment_text'][i] = train['comment_text'][i].lower()
    j = []
    for word in train['comment_text'][i].split():
        j.append(lemmatizer.lemmatize(word, pos="v"))
        train['comment_text'][i] = "".join(j)
        
x = train.drop(['toxic'],axis=1)
y = train['toxic']

naive = MultinomialNB()
tf_vec = TfidfVectorizer()

comment = train['comment_text']
x = tf_vec.fit_transform(comment)

x_train, x_test, y_train, y_test = train_test_split(x,y ,random_state=33)
naive.fit(x_train, y_train)

y_pred = naive.predict(x_test)

print(y_pred)

from sklearn.metrics import r2_score
R2 = r2_score(x_test, y_pred)
print(R2)
