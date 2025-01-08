from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
with open("BBDD Servicio Geologico/sismo_predictor.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=("GET", "POST"))
def index(): 
    
    # Obtener los datos enviados por el cliente
    data = request.get_json()
    
    # Asegúrate de que los datos coincidan con las columnas esperadas
    input_data = pd.DataFrame([data])  # Convertir a DataFrame
    
    # Predecir la intensidad del sismo
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
