from flask import Flask,render_template,request
import jsonify
import requests
import pickle
import numpy as np
import pandas
import string
import nltk



app=Flask(__name__)
loadedmodel=pickle.load(open('nlp.pkl','rb'))

@app.route("/" , methods=['GET'])
def Home():
    return  render_template('nlp.html')


def textprocessing(mess):
    cln=[]
    #removing punctuation
    for char in mess:
        
        if char not in string.punctuation:
            cln.append(char)
            
    
    
    #join the characters back
    cln="".join(cln)
    
    
    #filtering out the stopwords and returning the clean message
    cln=[word for word in cln.split() if word.lower() not in nltk.corpus.stopwords.words('English')]
    return cln   
    
@app.route("/predict" ,methods=['POST'])

def predict():
    Message=(request.form['Message'])
    Message=textprocessing(Message)

    print("Message:",Message)

    
    prediction=loadedmodel.predict(Message)[0]

    print(prediction)
    
    
    return render_template('nlp.html',prediction=str(prediction))
if __name__=='__main__' :
    app.run(debug=True)
    
