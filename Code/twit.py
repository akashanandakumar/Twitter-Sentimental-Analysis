import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv("twitter_train.csv") #10980 rows 12 cols
test = pd.read_csv("twitter_test.csv")

drop_cols = ['airline_sentiment_gold','name','tweet_id', 'retweet_count','tweet_created','user_timezone','tweet_coord','tweet_location']
train.drop(drop_cols, axis = 1, inplace=True)
test.drop(drop_cols, axis = 1, inplace=True)

stops = stopwords.words('english')
stops += list(punctuation)
stops += ['flight','airline','flights','AA']

abbreviations = {'ppl': 'people','cust':'customer','serv':'service','mins':'minutes','hrs':'hours','svc': 'service',
           'u':'you','pls':'please'}

train_index = train[~train.negativereason_gold.isna()].index
test_index = test[~test.negativereason_gold.isna()].index

for index, row in test.iterrows():
    tweet = row.text
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #remove links
    tweet = re.sub('@[^\s]+','',tweet) #remove usernames
    tweet = re.sub('[\s]+', ' ', tweet) #remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #replace #word with word
    tweet = tweet.strip('\'"') #trim tweet
    words = []
    for word in tweet.split(): 
        if word.lower() not in stops:
            if word in list(abbreviations.keys()):
                words.append(abbreviations[word])
            else:
                words.append(word.lower())
    tweet = " ".join(words)
    tweet = " %s %s" % (tweet, row.airline)
    row.text = tweet
    if index in test_index:
        row.text = " %s %s" % (row.text, row.negativereason_gold)

del train['negativereason_gold']
del test['negativereason_gold']

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

for index, row in train.iterrows():
    row.text = deEmojify(row.text)

for index, row in test.iterrows():
    row.text = deEmojify(row.text)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

for index, row in train.iterrows():
    words = row.text.split()
    new_words = []
    for word in words:
        if not hasNumbers(word):
            new_words.append(word)
    row.text = " ".join(new_words)
    
for index, row in test.iterrows():
    words = row.text.split()
    new_words = []
    for word in words:
        if not hasNumbers(word):
            new_words.append(word)
    row.text = " ".join(new_words)

train.head()

v = TfidfVectorizer(analyzer='word', max_features=3150, max_df = 0.8, ngram_range=(1,1))
train_features= v.fit_transform(train.text)
test_features=v.transform(test.text)

clf = LogisticRegression(C = 2.1, solver='liblinear', multi_class='auto')
clf.fit(train_features,train['airline_sentiment'])
pred = clf.predict(test_features)
with open('predictions_twitter.csv', 'w') as f:
    for item in pred:
        f.write("%s\n" % item)

clf = SVC(kernel="linear", C= 0.96 , gamma = 'scale')
# clf = SVC(C = 1000, gamma = 0.001)
clf.fit(train_features, train['airline_sentiment'])
pred = clf.predict(test_features)

with open('predictions_twitter2.csv', 'w') as f: #less accurate
    for item in pred:
        f.write("%s\n" % item)