from flask import Flask, redirect

app = Flask(__name__)


@app.route("/")
def home():
    return redirect("http://localhost:3000/streamlit")  # redirect to Streamlit app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # runs flask on port 5000
