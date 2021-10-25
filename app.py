from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS,cross_origin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing import sequence
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner(path='./models', file='model.pkl')
classes = learn.data.classes


# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


def predict_response(response):
    response = tokenizer.texts_to_sequences(response)
    response = sequence.pad_sequences(response, maxlen=48)
    probs = np.around(model.predict(response),decimals=2)
    pred = np.argmax(probs)
    #print(probs)
    #print(pred)
    if pred == 0:
        tag = 'Very Negative'
        tag_prob = probs[0,0]
        sent_prob = np.sum(probs[0,:2])
    elif pred == 1:
        tag = 'Negative'
        tag_prob = probs[0,1]
        sent_prob = np.sum(probs[0,:2])
    elif pred == 2:
        tag = 'Neutral'
        tag_prob = probs[0,2]
        sent_prob = probs[0,2]        
    elif pred == 3:
        tag = 'Positive'
        tag_prob = probs[0,3]
        sent_prob = np.sum(probs[0,3:])
    elif pred == 4:
        tag = 'Very Positive'
        tag_prob = probs[0,4]
        sent_prob = np.sum(probs[0,3:])
    return tag, tag_prob, sent_prob



# route for prediction
#@app.route('/predict', methods=['POST'])


if __name__ == "__main__":
    app.run(debug=True)