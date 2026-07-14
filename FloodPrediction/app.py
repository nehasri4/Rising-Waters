from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/flood_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["Rainfall"])
    cloud = float(request.form["CloudVisibility"])
    seasonal = float(request.form["SeasonalRainfall"])
    temperature = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])

    data = np.array([[rainfall, cloud, seasonal, temperature, humidity]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "⚠ HIGH FLOOD RISK"
    else:
        result = "✅ NO FLOOD RISK"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)