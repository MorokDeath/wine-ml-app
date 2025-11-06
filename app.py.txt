from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Загружаем обученную модель
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f'Предсказанное качество вина: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Ошибка: {e}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
