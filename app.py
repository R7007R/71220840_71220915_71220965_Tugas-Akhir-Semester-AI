from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

rf = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = int(request.form['time'])
    precipIntensity = float(request.form['precipIntensity'])
    precipProbability = float(request.form['precipProbability'])
    temperature = float(request.form['temperature'])
    apptemp = float(request.form['apptemp'])
    dewpoint = float(request.form['dewpoint'])
    humidity = float(request.form['humidity'])
    pressure = float(request.form['pressure'])
    wind_speed = float(request.form['wind_speed'])
    wind_gust = float(request.form['wind_gust'])
    wind_bearing = int(request.form['wind_bearing'])
    cloud_cover = float(request.form['cloud_cover'])
    uv_index = float(request.form['uv_index'])
    visibility = float(request.form['visibility'])
    ozone = float(request.form['ozone'])

    input_data = {
        "time": time,
        "precipIntensity": precipIntensity,
        "precipProbability": precipProbability,
        "temperature": temperature,
        "apptemp": apptemp,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "wind_bearing": wind_bearing,
        "cloud_cover": cloud_cover,
        "uv_index": uv_index,
        "visibility": visibility,
        "ozone": ozone
    }

    df_input_user = pd.DataFrame(input_data, index=[0])
    x_input_user = df_input_user.drop(["time"], axis=1)
    pred_input_user = rf.predict(x_input_user)

    prediction = pred_input_user[0]
    prediction_image = f"{prediction}.png"

    result = 'Hasil prediksi cuaca adalah: ' + prediction

    return render_template('index.html', prediction=result, prediction_image=prediction_image)

if __name__ == '__main__':
    app.run(debug=True)
