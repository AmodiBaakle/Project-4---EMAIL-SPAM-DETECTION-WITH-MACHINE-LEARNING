#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 


# In[81]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[117]:


data = pd.read_csv('spam2.csv')
print(data.head())


# In[48]:


#check for null values 
print(data.isnull().sum())


# In[49]:


#rows X columns 
data.shape


# In[118]:


data.columns= ["Category","Msg"]
print(data.head())


# In[120]:


#spam = 0 , ham = 1
data.loc[data['Category']=='spam', 'Category']=0
data.loc[data['Category']=='ham','Category']=1


# In[121]:


print(data)


# In[126]:


#separating the data text and label
X = data['Msg']
Y = data['Category']


# In[125]:


print(X)


# In[127]:


print(Y)


# In[128]:


X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size=0.2, random_state=3)


# In[129]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[130]:


#transform the text data to feature vectors that can be used as input to the logistics regression 


# In[101]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english',lowercase='True')


# In[135]:


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test) 


# In[136]:


#convert Y_train and Y_test
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[137]:


print(X_train)


# In[142]:


print(Y_train.head(20))


# In[139]:


print(X_train_features.shape)


# In[143]:


#model training 
model = LogisticRegression()

#passing the training data to the model
model.fit(X_train_features,Y_train)


# In[144]:


#model evaluation : trainiong data 
perdition_on_training_data = model.predict(X_train_features)
accurcy_on_training_data = accuracy_score(Y_train,perdition_on_training_data)


# In[150]:


print("Accuracy on training data = ",accurcy_on_training_data*100 )


# In[147]:


#test data
perdition_on_testing_data = model.predict(X_test_features)
accurcy_on_testing_data = accuracy_score(Y_test,perdition_on_testing_data)


# In[149]:


print("Accuracy on testing data = ",accurcy_on_testing_data*100 )


# In[173]:


input_mail = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]
print(type(input_mail))
#conversion 


# In[168]:


input_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_features)
#print(prediction)
#print(type(prediction))
if prediction[0]==1:
    print("The mail is a ham mail")
else:
    print("The mail is a spam mail")


# In[176]:


mail = [input("Mail : ")]
mail_features = feature_extraction.transform(mail)
predict_mail = model.predict(mail_features)
if predict_mail[0]==1:
    print("The mail is a ham mail")
else : 
    print("The mail is a spam mail")







