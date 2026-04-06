# from utils import db_connect
# engine = db_connect()

# your code here
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('wine_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Recoger valores
        features = [float(x) for x in request.form.values()]
        
        # 2. Nombres EXACTOS de las columnas del dataset original
        column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol']
        
        # 3. Crear un DataFrame de Pandas (como lo espera el modelo)
        df_features = pd.DataFrame([features], columns=column_names)
        
        # 4. Predecir
        prediction = model.predict(df_features)
        output = prediction[0]
        
        return render_template('index.html', prediction_text=f'La calidad predicha de este vino es: {output} (escala 0-10)')
    
    return render_template('index.html', prediction_text=None)

if __name__ == "__main__":
    app.run(port=8000, debug=True)