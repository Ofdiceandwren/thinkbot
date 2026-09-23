from flask import Flask, render_template, request
import markdown

from thinkbot import run_thinkbot

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        question = request.form.get("question")
        mode = request.form.get("mode")
        print("Mode:", mode, flush=True)
        print("Question:", question, flush=True)
        response = run_thinkbot(mode, question)
        response = markdown.markdown(response)


    return render_template(
    "index.html",
    response=response if request.method == "POST" else None,
    question=question if request.method == "POST" else "",
    selected_mode=mode if request.method == "POST" else "Answer as Chatbot"
)


if __name__ == "__main__":
    app.run(debug=True)