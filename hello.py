
# coding: utf-8

# In[1]:


import sys, os, re, csv, codecs, numpy as np
import pandas as pd
import json
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask import Flask, url_for, request, json, jsonify
import tensorflow as tf

'''
tf.app.flags.DEFINE_integer(
    "doc_vocab_size", 232081, "Document vocabulary size.")
tf.app.flags.DEFINE_integer(
    "sum_vocab_size", 232081, "Summary vocabulary size.")
'''

filename = "tokenizer_old(1).sav"
tokenizer = ""
# load the model from disk
# with open(filename,'rb') as handle:
tokenizer = pickle.load(open(filename,'rb'))


# In[3]:


# Load pre-trained models
filename = "my_model.h5"

# returns a compiled model
# identical to the previous one
model = load_model(filename)


# In[4]:



app = Flask(__name__)

# This function will take a json file
@app.route('/postjson', methods=['POST'])
def postJsonHandler():
    data = request.data
    data = data.decode("utf-8")
    data = json.loads(data)

    print("I'm getting called")

    if not data:
        raise Exception("The list is empty :(")
    else:

    # Convert json to pandas dataframe
    pandasData = pd.read_json(data)

    #

        listOfPosts = []
        for i in data:
            if i is str:
                x = json.loads(i)
            else:
                x = i
            if not x.get('content'):
                raise Exception("Error! The data might not be in the correct format")
            listOfPosts.append(x.get('content'))


        # Tokenize the data
        token = tokenizeData(listOfPosts)
        listOfResults = []

        # Predict each label and jsonify it
        prediction = predictText(token).tolist()
        counter = 0
        for i in prediction:
            classify_results = {
                "toxic":i[0],
                "severe_toxic":i[1],
                "obscene":i[2],
                "threat":i[3],
                "insult":i[4],
                "identity_hate":i[5]
            }

            result = {}
            result['ni'] = data[counter]
            result['classify_results'] = classify_results
            listOfResults.append(result)
            counter = counter + 1


        # Format results into data to be returned by the function


        #token = json.dumps(token)
        print(jsonify({'result':listOfResults}))
        return jsonify({'result':listOfResults})

#with app.test_request_context():
	#print(url_for('hello'))

def tokenizeData(X):
    # Transform data
    token = tokenizer.texts_to_sequences(X)
    token = pad_sequences(token, maxlen=200)
    return token

def predictText(X):
    # Predict probabilities for labels
    prediction = model.predict(np.array(X),batch_size=1000)
    return prediction

def add_features(df):

    df['comment_text'] = df['comment_text'].apply(lambda x:str(x))
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.comment_text.str.count('\S+')
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df

# PARAMETERS
#   predictions - <class 'list'> - This is a list of the predictions that are associated with each variable. These will
#        used to determine whether this text fits a label or not based on the specified threshold
#   threshold - <class 'float'> - This is the
def binaryPredictions(predictions,threshold=0.5):
    counter = 0
    for i in predictions:
        if i > threshold:
            predictions[counter] = 1
        else:
            predictions[counter] = 0
        counter += 1

    return predictions




# In[5]:


text = "Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.,"


# In[6]:


import pandas as pd


# In[7]:


token = tokenizer.texts_to_sequences([text])


# In[8]:


token[0][-10:]


# In[9]:


pad_sequences(token, maxlen=200,padding="post")


# In[10]:


pad_sequences(token, maxlen=200).shape


# In[11]:


model.predict(pad_sequences(token, maxlen=200))


# In[12]:


if __name__ == "__main__":
    app.run()


# In[13]:
