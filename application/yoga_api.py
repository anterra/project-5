import pickle
import numpy as np
import pandas as pd
import keras
from keras import Model


def get_class(pose_dict):
    """ Creates list of pose names from the pose drop-downs the user selected. """
    list_of_inputs = list(pose_dict.values())
    list_of_inputs = [pose.replace("-", " ") for pose in list_of_inputs]
    list_of_poses = [i for i in list_of_inputs if i != '']
    return list_of_poses


def get_class_info(pose_dict):

    list_of_inputs = list(pose_dict.values())
    list_of_inputs = [pose.replace("-", " ") for pose in list_of_inputs]
    list_of_poses = [i for i in list_of_inputs if i != '']

    if list_of_poses:
        class_df = df.loc[df["Pose Name"] == list_of_poses[0]]
        for pose in list_of_poses[1:]:
            row = df.loc[df["Pose Name"] == pose]
            class_df = pd.concat([class_df, row])

        class_length = len(class_df)
        class_df = class_df.iloc[:, 3:]
        counts = class_df.sum(axis=0)
        ratios = counts / class_length
        ratios["Class Length"] = class_length

        return dict(ratios)


def make_prediction(feature_dict):
    x_input = []
    for name in model.feature_names:
        x_input_ = float(feature_dict.get(name, 0))
        x_input.append(x_input_)

    pred_probs = model.predict_proba([x_input]).flat

    probs = []
    for index in np.argsort(pred_probs)[::-1]:
        prob = {
            "name": model.target_names[index],
            "prob": round(pred_probs[index], 5)
        }
        probs.append(prob)

    return x_input, probs


# checks that prediction code runs properly from terminal
if __name__ == "__main__":
    from pprint import pprint
    print("Checking to see what setting all params to 0 predicts")
    features = {f: "0" for f in feature_names}
    print("Features are")
    pprint(features)

    x_input, probs = make_prediction(features)
    print(f"Input values: {x_input}")
    print("Output probabilities")
    pprint(probs)
