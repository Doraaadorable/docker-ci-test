from flask import Flask, request, jsonify
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import socket
import os
import psycopg2

app = Flask(__name__)

# Train the model once when the application starts.
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model trained. Test accuracy: {accuracy:.3f}")


@app.route("/")
def home():
    return f"<h1>Hello</h1><p>Model test accuracy: {accuracy:.1%}</p>"


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        test_accuracy=round(accuracy, 3),
        pod=socket.gethostname(),
    )


@app.route("/sample/<int:i>")
def sample(i):
    return jsonify(
        pixels=X_test[i].tolist(),
        label=int(y_test[i]),
    )


@app.route("/predict", methods=["POST"])
def predict():
    pixels = request.get_json()["pixels"]
    return jsonify(prediction=int(model.predict([pixels])[0]))


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        dbname=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )


@app.route("/visits")
def visits():
    con = get_conn()
    try:
        with con, con.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS visits "
                "(id SERIAL PRIMARY KEY, at TIMESTAMP DEFAULT now())"
            )
            cur.execute("INSERT INTO visits DEFAULT VALUES")
            cur.execute("SELECT COUNT(*) FROM visits")
            n = cur.fetchone()[0]
    finally:
        con.close()
    return jsonify(visits=n)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)