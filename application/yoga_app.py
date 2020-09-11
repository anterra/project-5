import flask
from flask import request
from yoga_api import peak_poses, forward_model, backward_model, embeddings, tokenizer, tokenizer2, generate_class, get_peak_pose

# initialize the app
app = flask.Flask(__name__)


@app.route("/")
def create_yoga_class():
    return flask.render_template("home_page.html",
                                 peak_poses=peak_poses)


@app.route("/take_class", methods=["POST", "GET"])
def predict():

    peak_pose = get_peak_pose(request.form)
    first_half = generate_class(
        backward_model, tokenizer2, embeddings, peak_pose, 40)
    second_half = generate_class(
        forward_model, tokenizer, embeddings, peak_pose, 40)[::-1]
    yoga_class = first_half + second_half

    if request.method == 'POST':
        result = yoga_class
        return flask.render_template("yoga_class_page.html",
                                     result=result,
                                     peak_pose=peak_pose,
                                     )


# start the server, continuously listen to requests

# for local development:
if __name__ == "__main__":
    app.run(debug=True)

# for public web serving:
app.run(host="0.0.0.0")
