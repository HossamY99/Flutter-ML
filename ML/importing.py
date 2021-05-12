# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:51:47 2021

@author: hossa
"""


import json

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys
print(sys.executable)
#from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import re
import collections
from wordcloud import STOPWORDS
from scipy.sparse import csr_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from scipy.sparse import hstack
import pickle

















#print(pd.__version__)
train_data = pd.read_table('train.tsv',sep='\t')
train_data=train_data.head(500000)
print(train_data.shape)
print(train_data.isnull().sum())




train_data = train_data[train_data['price'] > 0].reset_index(drop=True)
train_data,cv_data=train_test_split(train_data,test_size=0.1,random_state=42)
print("cv data: /n")



train=train_data.copy()
val=cv_data.copy()


def handle_category(data):
  """this function splits the category_name into further three sub_categories."""
  cat1=[]
  cat2=[]
  cat3=[]
  i=0
  for row in data:
    try:
      categories=row.split('/')
    except:
      categories=['','','']
    cat1.append(categories[0])
    cat2.append(categories[1])
    cat3.append(categories[2])
    i+=1
  return cat1,cat2,cat3


#c1,c2,c3=handle_category(train_data['category_name'])
#train_data['sub_category1']=c1
#train_data['sub_category2']=c2
#train_data['sub_category3']=c3
c1,c2,c3=handle_category(cv_data['category_name'])
cv_data['sub_category1']=c1
cv_data['sub_category2']=c2
cv_data['sub_category3']=c3

#train_data['item_description'].fillna(value='No description given',inplace=True)
#train_data['brand_name'].fillna(value='Not known',inplace=True)
#train_data.isnull().sum()


cv_data['item_description'].fillna(value='No description given',inplace=True)
cv_data['brand_name'].fillna(value='Not known',inplace=True)
cv_data.isnull().sum()




cars = {'name': ['Brandy Melville Off Shoulder Crop Top'],
        'item_condition_id': [2],
        'category_name': ['Women/Tops & Blouses/Blouse'],
        'brand_name': ['Brandy Melville'],
        #'price': [21.0],
        'shipping': [0],
        'item_description': ['Brandy Melville Off Shoulder Crop Top. One Size Fits All. EUC. 3/4 sleeves. 100% Rayon. Actual color: Navy Blue and cream in stripes.']
        }

df = pd.DataFrame(cars, columns = ['name','item_condition_id','category_name','brand_name','shipping','item_description'])

print (df)

#test_data=pd.read_table('test.tsv')
test_data=df
print(test_data.head(1))
test_data=test_data.head(200)
test=test_data.copy()


print("shape of the test data: ",test_data.shape)
test_data.isnull().sum()

print("Number of Nan values in category_name: {}%".format((test_data['category_name'].isnull().sum()/test_data.shape[0])*100))
print("Number of Nan values in brand_name: {}%".format((test_data['brand_name'].isnull().sum()/test_data.shape[0])*100))
print("Number of Nan values in item description: {}%".format((test_data['item_description'].isnull().sum()/test_data.shape[0])*100))


c1,c2,c3=handle_category(test_data['category_name'])
test_data['sub_category1']=c1
test_data['sub_category2']=c2
test_data['sub_category3']=c3


test_data['brand_name'].fillna(value='Not known',inplace=True)
test_data['item_description'].fillna(value='No description given',inplace=True)
test_data.isnull().sum()



stopwords=set(stopwords.words('english'))



def stopwords_count(data):
  """this function counts the number of stopwords in each of the item_description"""
  count_stopwords=[]
  for i in tqdm(data['item_description']):
    count=0
    for j in i.split(' '):
      if j in stopwords: count+=1  #finding if the word is present in the nltk stopwords or not
    count_stopwords.append(count)
  return count_stopwords


#train_data['count_stopwords']=stopwords_count(train_data)
cv_data['count_stopwords']=stopwords_count(cv_data)
test_data['count_stopwords']=stopwords_count(test_data)





#train_data['count_stopwords'].describe()


# https://stackoverflow.com/a/47091490/4084039
def decontracted(phrase):
    """this function removies shorthands for the textual data..."""
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# https://gist.github.com/sebleier/554280
def text_preprocessing(data):
  """this function performs preprocessing the item_description """
  preprocessed_total = []
  for sentance in tqdm(data['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)  #removing stop words
    preprocessed_total.append(sent.lower().strip())
  return preprocessed_total



#train_data['item_description']=text_preprocessing(train_data)
cv_data['item_description']=text_preprocessing(cv_data)
test_data['item_description']=text_preprocessing(test_data)

stopwords=set(STOPWORDS)
#word_cloud = WordCloud(width = 600, height = 600,background_color ='white', stopwords=stopwords,min_font_size = 10).generate("1 ".join(train_data['item_description']))
#plt.figure(figsize = (15, 10))
#plt.imshow(word_cloud)
#plt.axis('off')                                             
#plt.show()




#word_count={}
#for sentence in tqdm(train_data['item_description']):
#    for word in sentence.split(' '):
#        if len(word)>=3:  #taking words which are of length>=3
#            if word not in word_count:
#                word_count[word]=1  #if word not present in dict assigning it to 1
#            else:
#                word_count[word]+=1 #else incrementing it by 1


#n_print=25
#word_counter = collections.Counter(word_count)
#words=[]
#counter=[]
#for word, count in word_counter.most_common(n_print):
#    words.append(word)
#    counter.append(count)


#plt.figure(figsize=(10,7))
#sns.barplot(counter,words)
#plt.title("25 Most Frequent Words in Item-Description")
#plt.xlabel('Frequency')
#plt.ylabel('Words')
#plt.show()



def description_length(data):
  """this function finds the length of the description basing on spaces in the statement"""
  description_length=[]
  for i in data['item_description']:
    description_length.append(len(i.split(' '))) #splitting statement using spaces and finding length of it
  return description_length




print("processing item_description in train_data...")
#train_data['description_length']=description_length(train_data)
cv_data['description_length']=description_length(cv_data)
#print(train_data.iloc[100]['item_description'],train_data.iloc[100]['description_length'])
#print("="*100)
#print("processing item_description in test_data...")
test_data['description_length']=description_length(test_data)
#print(test_data.iloc[100]['item_description'],test_data.iloc[100]['description_length'])





#train_data['description_length'].describe()


def branded(data):
   """this function assigns a value 1 if a product has brand_name else 0"""
   is_branded=[]
   for i in data['brand_name']:
      if i=='Not known': is_branded.append(0) #if it is a Nan value i.e.. unknown brand make it as 0.
      else: is_branded.append(1)
   return is_branded
#train_data['is_branded']=branded(train_data)
cv_data['is_branded']=branded(cv_data)
test_data['is_branded']=branded(test_data)


def sentiment_analysis(data):
   """this function performs sentiment score analysis of each datapoint"""
   sentiment_score = SentimentIntensityAnalyzer()
   sentiment = []
   for sentence in tqdm(data):
       sentiment.append(sentiment_score.polarity_scores(sentence))
   return sentiment



#training_sentiment_score=sentiment_analysis(train_data['item_description']) 
cv_sentiment_score=sentiment_analysis(cv_data['item_description'])
testing_sentiment_score=sentiment_analysis(test_data['item_description'])

def splitting_sentiment(sentiment_score):
  """this function splits sentiment analysis score into four further features ie positive,negative,compound and neutral"""
  positive=[]
  negative=[]
  neutral=[]
  compound=[]
  for i in sentiment_score:
    positive.append(i['pos'])
    negative.append(i['neg'])
    neutral.append(i['neu'])
    compound.append(i['compound'])
  return positive,negative,neutral,compound




#print("Training Data Sentiment Analysis: ")
#pos,neg,neu,comp=splitting_sentiment(training_sentiment_score)
#train_data['positive']=pos
#train_data['negative']=neg
#train_data['neutral']=neu
#train_data['compound']=comp
#print(train_data.iloc[50]['item_description'])
#print(training_sentiment_score[50])


print("CV Data Sentiment Analysis: ")
pos,neg,neu,comp=splitting_sentiment(cv_sentiment_score)
cv_data['positive']=pos
cv_data['negative']=neg
cv_data['neutral']=neu
cv_data['compound']=comp
#print(cv_data.iloc[50]['item_description'])
#print(cv_sentiment_score[50])





print("Testing Data Sentiment Analysis: ")
pos,neg,neu,comp=splitting_sentiment(testing_sentiment_score)
test_data['positive']=pos
test_data['negative']=neg
test_data['neutral']=neu
test_data['compound']=comp
#print(test_data.iloc[50]['item_description'])
print(testing_sentiment_score[0])




#train_data['target']=np.log(np.array(train_data['price'].values)+1)
cv_data['target']=np.log(np.array(cv_data['price'].values)+1)
#train_data.drop(['train_id','category_name'],axis=1,inplace=True)
#cv_data.drop(['train_id','category_name'],axis=1,inplace=True)






#countvectorizer=CountVectorizer().fit(train_data['sub_category1'])     
#with open('countv1', 'wb') as fout:
#    pickle.dump(countvectorizer, fout)

with open('countv1','rb') as f:
    countv1=pickle.load(f)

                #fitting
#bow_cat1_train=countv1.transform(train_data['sub_category1'])
bow_cat1_cv=countv1.transform(cv_data['sub_category1'])
bow_cat1_test=countv1.transform(test_data['sub_category1'])
#print("After Vectorization of sub category1 feature: ")
#print(bow_cat1_train.shape)
#print(bow_cat1_cv.shape)
#print(bow_cat1_test.shape)
#print("Some Features are: ")
#print(countvectorizer.get_feature_names())
#print("="*125)
#countvectorizer=CountVectorizer().fit(train_data['sub_category2'])   #fitting
#with open('countv2', 'wb') as fout:
#    pickle.dump(countvectorizer, fout)

with open('countv2','rb') as f:
    countv2=pickle.load(f)
#bow_cat2_train=countv2.transform(train_data['sub_category2'])
bow_cat2_cv=countv2.transform(cv_data['sub_category2'])
bow_cat2_test=countv2.transform(test_data['sub_category2'])
#print("After Vectorization of sub category2 feature: ")
#print(bow_cat2_train.shape)
#print(bow_cat2_cv.shape)
#print(bow_cat2_test.shape)
#print("Some Features are: ")
#print(countvectorizer.get_feature_names()[50:75])
#print("="*125)
#countvectorizer=CountVectorizer().fit(train_data['sub_category3'])   
#with open('countv22', 'wb') as fout:
#    pickle.dump(countvectorizer, fout)

with open('countv22','rb') as f:
    countv22=pickle.load(f)
#fitting
#bow_cat3_train=countv22.transform(train_data['sub_category3'])
bow_cat3_cv=countv22.transform(cv_data['sub_category3'])
bow_cat3_test=countv22.transform(test_data['sub_category3'])
#print("After Vectorization of sub category3 feature: ")
#print(bow_cat3_train.shape)
#print(bow_cat3_cv.shape)
#print(bow_cat3_test.shape)
#print("Some Features are: ")
#print(countvectorizer.get_feature_names()[50:75])
#print("="*125)
#countvectorizer=CountVectorizer().fit(train_data['brand_name'])  #fitting
#with open('countv3', 'wb') as fout:
#    pickle.dump(countvectorizer, fout)

with open('countv3','rb') as f:
    countv3=pickle.load(f)

#bow_brand_train=countv3.transform(train_data['brand_name'])
bow_brand_cv=countv3.transform(cv_data['brand_name'])
bow_brand_test=countv3.transform(test_data['brand_name'])
#print("After Vectorization of brand_name feature: ")
#print(bow_brand_train.shape)
#print(bow_brand_cv.shape)
#print(bow_brand_test.shape)
#print("Some Features are: ")
#print(countvectorizer.get_feature_names()[50:75])
#print("="*125)





#countvectorizer=CountVectorizer(min_df=10).fit(train_data['name'])  #fitting
#with open('countv5', 'wb') as fout:
#    pickle.dump(countvectorizer, fout)

with open('countv5','rb') as f:
    countv5=pickle.load(f)

#bow_name_train=countv5.transform(train_data['name'])
bow_name_cv=countv5.transform(cv_data['name'])
bow_name_test=countv5.transform(test_data['name'])
#print("After Vectorization of brand_name feature: ")
#print(bow_name_train.shape)
#print(bow_name_cv.shape)
#print(bow_name_test.shape)
#print("Some Features are: ")
#print(countvectorizer.get_feature_names()[10000:10025])



#tfidfvectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=10,\
#                                max_features=5000).fit(train_data['item_description']) #fitting
    
#with open('tf1', 'wb') as fout:
#    pickle.dump(tfidfvectorizer, fout)

with open('tf1','rb') as f:
    tf1=pickle.load(f)
#tfidf_description_train=tf1.transform(train_data['item_description'])
tfidf_description_cv=tf1.transform(cv_data['item_description'])
tfidf_description_test=tf1.transform(test_data['item_description'])
#print("After Vectorization of item description feature: ")
#print(tfidf_description_train.shape)
#print(tfidf_description_cv.shape)
#print(tfidf_description_test.shape)
#print("Some Features are: ")
#print(tfidfvectorizer.get_feature_names()[3025:3050])  #getting 25 random features.



#scaler=StandardScaler().fit(np.array(train_data['positive']).reshape(-1,1))   #fitting

#with open('s1', 'wb') as fout:
#    pickle.dump(scaler, fout)

with open('s1','rb') as f:
    s1=pickle.load(f)
#positive_train = s1.transform(np.array(train_data['positive']).reshape(-1,1))
positive_cv = s1.transform(np.array(cv_data['positive']).reshape(-1,1))
positive_test = s1.transform(np.array(test_data['positive']).reshape(-1,1))
#print(positive_train[50:55].reshape(1,-1)[0])    #printing 5 random postive sentiment scores 
#print("After Preprocessing of positive sentiment score:")
#print(positive_train.shape)
#print(positive_cv.shape)
#print(positive_test.shape)
#print("="*125)





#scaler2 = StandardScaler().fit(np.array(train_data['negative']).reshape(-1,1))  #fitting
#scaler3 = StandardScaler().fit(np.array(train_data['neutral']).reshape(-1,1))   #fitting
#scaler4 = StandardScaler().fit(np.array(train_data['compound']).reshape(-1,1))  #fitting
#scaler5 = StandardScaler().fit(np.array(train_data['description_length']).reshape(-1,1))  #fitting
#scaler6 = StandardScaler().fit(np.array(train_data['count_stopwords']).reshape(-1,1))   #fitting


#with open('scalers2to6', 'wb') as fout:
#    pickle.dump((scaler2,scaler3,scaler4,scaler5,scaler6), fout)

with open('scalers2to6','rb') as f:
    s2,s3,s4,s5,s6=pickle.load(f)


#negative_train=s2.transform(np.array(train_data['negative']).reshape(-1,1))
negative_cv=s2.transform(np.array(cv_data['negative']).reshape(-1,1))
negative_test=s2.transform(np.array(test_data['negative']).reshape(-1,1))
#print(negative_train[25:30].reshape(1,-1)[0])    #printing 5 random negative sentiment score
#print("After Preprocessing of negative sentiment score:")
#print(negative_train.shape)
#print(negative_cv.shape)
#print(negative_test.shape)
#print("="*125)

#neutral_train=s3.transform(np.array(train_data['neutral']).reshape(-1,1))
neutral_cv=s3.transform(np.array(cv_data['neutral']).reshape(-1,1))
neutral_test=s3.transform(np.array(test_data['neutral']).reshape(-1,1))
#print(neutral_train[5:10].reshape(1,-1)[0])     #printing 5 random neutral sentiment score
#print("After Preprocessing of neutral sentiment score:")
#print(neutral_train.shape)
#print(neutral_cv.shape)
#print(neutral_test.shape)
#print("="*125)

#compound_train=s4.transform(np.array(train_data['compound']).reshape(-1,1))
compound_cv=s4.transform(np.array(cv_data['compound']).reshape(-1,1))
compound_test=s4.transform(np.array(test_data['compound']).reshape(-1,1))
#print(compound_train[35:40].reshape(1,-1)[0])   #printing 5 random compound sentiment score
#print("After Preprocessing of compound sentiment score:")
#print(compound_train.shape)
#print(compound_cv.shape)
#print(compound_test.shape)
#print("="*125)

#length_train=s5.transform(np.array(train_data['description_length']).reshape(-1,1))
length_cv=s5.transform(np.array(cv_data['description_length']).reshape(-1,1))
length_test=s5.transform(np.array(test_data['description_length']).reshape(-1,1))
#print(length_train[1:5].reshape(1,-1)[0])       #printing 5 random description lengths
#print("After Preprocessing of description length:")
#print(length_train.shape)
#print(length_cv.shape)
#print(length_test.shape)
#print("="*125)

#stopword_train=s6.transform(np.array(train_data['count_stopwords']).reshape(-1,1))
stopword_cv=s6.transform(np.array(cv_data['count_stopwords']).reshape(-1,1))
stopword_test=s6.transform(np.array(test_data['count_stopwords']).reshape(-1,1))
#print(stopword_train[15:20].reshape(1,-1)[0])   #printing 5 random stopwords count
#print("After Preprocessing of count_stopwords feature:")
#print(stopword_train.shape)
#print(stopword_cv.shape)
#print(stopword_test.shape)







#https://stackoverflow.com/questions/36285155/pandas-get-dummies

#features_train = csr_matrix(pd.get_dummies(train_data[['item_condition_id', 'shipping','is_branded']],sparse=True).values)
features_cv = csr_matrix(pd.get_dummies(cv_data[['item_condition_id', 'shipping','is_branded']],sparse=True).values)
features_test = csr_matrix(pd.get_dummies(test_data[['item_condition_id', 'shipping','is_branded']],sparse=True).values)
#print("shape",features_train.shape)
#print(features_train)
print(features_cv.shape)
print(features_test.shape)




#https://stackoverflow.com/questions/43018711/about-numpys-concatenate-hstack-vstack-functions

#X_train=hstack((bow_cat1_train,bow_cat2_train,bow_cat3_train,bow_brand_train,bow_name_train,tfidf_description_train,positive_train,negative_train,neutral_train,compound_train,features_train,length_train,stopword_train)).tocsr()
X_cv=hstack((bow_cat1_cv,bow_cat2_cv,bow_cat3_cv,bow_brand_cv,bow_name_cv,tfidf_description_cv,positive_cv,negative_cv,neutral_cv,compound_cv,features_cv,length_cv,stopword_cv)).tocsr()
X_test=hstack((bow_cat1_test,bow_cat2_test,bow_cat3_test,bow_brand_test,bow_name_test,tfidf_description_test,positive_test,negative_test,neutral_test,compound_test,features_test,length_test,stopword_test)).tocsr()
#print("Shape of train data: ",X_train.shape) #train
print("Shape of cv data: ",X_cv.shape)   #cv
print("Shape of test data: ",X_test.shape)   #test
print(X_cv)


lasso = Lasso(alpha=0.00001,fit_intercept=False)
print("Model is fitting!!!")
#lasso.fit(X_train, train_data['target'])

#with open('model_pickle','wb') as f:
#    pickle.dump(lasso,f)


with open('model_pickle','rb') as f:
    mod=pickle.load(f)
    
print("imp pred")

#print(mod.predict(X_train))
print("done")
#ytrain_predict=lasso.predict(X_train)
#print("Real\n",train_data.head())
#print("target\n",train_data['target'])
#print(ytrain_predict)
ycv_predict=mod.predict(X_cv)
print("predictions: ",ycv_predict)
#train_ = np.sqrt(mean_squared_error(train_data['target'], ytrain_predict))
cv_=np.sqrt(mean_squared_error(cv_data['target'],ycv_predict))
print("Lasso Regression with alpha = {} RMSLE on train is {} RMSLE on cv is {}".format(1e-06,1,cv_))



#print("train data head", train_data.head(1).price)



#'con':2,
#                    'cat':'Women/Tops & Blouses/Blouse',
#                      'brand':'Brandy Melville',
#                    'price':21.0,
#                     'shipping':0,
#                     'desc':'Brandy Melville Off Shoulder Crop Top. One Size Fits All. EUC. 3/4 sleeves. 100% Rayon. Actual color: Navy Blue and cream in stripes.'






ycv_lasso=mod.predict(X_cv)
ytesting=mod.predict(X_test[0])

#print(X_test[0])




#print(X_train[0])

y1=mod.predict(X_cv[0])
print("ytesting",ytesting)
print("y1:",y1)

ytest_lasso=mod.predict(X_test)
res = np.exp(ytest_lasso[0])-1
print(res)
ytest_lasso=mod.predict(X_cv[0])
res = np.exp(ytest_lasso[0])-1

print(res)
#print(X_cv[0])