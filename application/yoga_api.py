import pickle
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import Model
from tensorflow.keras.models import load_model


f = open("../word_embeddings.pkl", "rb")
embeddings = pickle.load(f)
f.close()

f = open("../tokenizer.pkl", "rb")
tokenizer = pickle.load(f)
f.close()

f = open("../tokenizer2.pkl", "rb")
tokenizer2 = pickle.load(f)
f.close()

f = open("../peak_poses.pkl", "rb")
peak_poses = pickle.load(f)
f.close()

forward_model = load_model("../forward_model.h5")
backward_model = load_model("../backward_model.h5")


def get_peak_pose(user_selection):
    peak_pose = user_selection
    return peak_pose


def generate_class(model, tokenizer, word_embedding, peak_pose, max_length):
    # generate seed text of 3 poses (as LSTM was trained on) from the input desired peak pose, by randomly selecting two of the most similar poses to the peak.
    seed_text = [peak_pose, embeddings.most_similar(peak_pose, topn=10)[np.random.choice(
        range(10))][0], embeddings.most_similar(peak_pose, topn=10)[np.random.choice(range(10))][0]]
    in_text = seed_text

    # create yoga class, and make sure it includes the user's desired peak pose
    yoga_class = list()
    yoga_class.append(peak_pose.lower())

    while True:
        encoded = tokenizer.texts_to_sequences([in_text])

        # select next pose based on models probability distribution
        prediction_output = (model.predict(encoded))
        yhat = np.random.choice(
            len(prediction_output[0]), p=prediction_output[0])

        out_word = ""
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break

        # append pose to current class, and update input text
        # also add clarification text for user to repeat the pose on the other side (if its an imbalanced pose) if given two of the same pose in a row
        if out_word != "":
            if out_word == yoga_class[-1]:
                in_text.append(out_word)
                out_word += ", repeat other side"
                yoga_class.append(out_word)
            else:
                yoga_class.append(out_word)
                in_text.append(out_word)
        if out_word in ["--", "corpse pose"]:
            break

        # if sequence gets too long without converging to a natural ending, start over.
        if len(yoga_class) == max_length:
            in_text = seed_text
            yoga_class = [peak_pose.lower()]
    return yoga_class


# check that code runs properly from terminal
if __name__ == "__main__":
    from pprint import pprint
