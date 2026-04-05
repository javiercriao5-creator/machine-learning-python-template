from utils import db_connect
engine = db_connect()

# your code here
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Como app.py y wine_model.pkl están en la misma carpeta (src), esto funcionará perfecto
model = pickle.load(open('wine_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Recopilamos los 11 valores químicos del formulario
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Hacemos la predicción
        prediction = model.predict(final_features)
        output = prediction[0]
        
        return render_template('index.html', prediction_text=f'La calidad predicha de este vino es: {output} (escala 0-10)')
    
    return render_template('index.html', prediction_text=None)

if __name__ == "__main__":
    app.run(port=8000, debug=True)