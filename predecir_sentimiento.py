import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string

"""
Interfaz de predicción con estilo minimalista tipo ChatGPT
----------------------------------------------------------
Este script carga el modelo entrenado, junto con su capa TextVectorization,
y permite introducir una frase para predecir si es Negative, Neutral o Positive,
con un estilo visual moderno en blanco y gris.
"""

# Carga el modelo en formato TF o .keras
# Ajusta la ruta según tu caso:
# modelo = tf.keras.models.load_model("modelo_sentimiento")
# modelo = tf.keras.models.load_model("modelo_sentimiento.keras")

# Ejemplo genérico, asumiendo carpeta "modelo_sentimiento"
modelo = tf.keras.models.load_model("modelo_sentimiento")

# Mapeo de índices a etiquetas
etiquetas = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def predecir_sentimiento(model, texto):
    """ Recibe un texto y devuelve la clase de sentimiento."""
    entrada = [texto]
    probabilidades = model.predict(entrada)
    indice_pred = np.argmax(probabilidades, axis=1)[0]
    return etiquetas[indice_pred]

app = Flask(__name__)

# Plantilla HTML con estilo minimalista en blanco y gris, inspirada en la interfaz de ChatGPT
html_template = """
<!DOCTYPE html>
<html lang=\"es\">
<head>
    <meta charset=\"utf-8\"/>
    <title>Clasificador de Sentimientos</title>
    <style>
        /* Estilos base para emular estilo limpio (blanco y gris) tipo ChatGPT */
        body {
            background-color: #F7F7F8;
            color: #3D3D3D;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, \"Open Sans\", \"Helvetica Neue\", sans-serif;
        }
        header {
            background-color: #FFFFFF;
            border-bottom: 1px solid #E0E0E0;
            padding: 1rem;
            text-align: center;
        }
        h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .container {
            max-width: 600px;
            margin: 2rem auto;
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            padding: 2rem;
        }
        label {
            display: inline-block;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            resize: vertical;
            padding: 0.5rem;
            font-size: 1rem;
            border: 1px solid #CCC;
            border-radius: 4px;
            outline: none;
        }
        textarea:focus {
            border-color: #B8B8B8;
        }
        .btn {
            display: inline-block;
            background-color: #3E3F42;
            color: #FFF;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            margin-top: 1rem;
        }
        .btn:hover {
            background-color: #1E1F21;
        }
        .result {
            margin-top: 1.5rem;
            background-color: #F2F2F2;
            padding: 1rem;
            border-radius: 4px;
        }
        .result p {
            margin: 0.3rem 0;
        }
    </style>
</head>
<body>

<header>
    <h1>Clasificador de Sentimientos</h1>
</header>

<div class=\"container\">
    <form method=\"POST\" style=\"margin-bottom: 1rem;\">
        <label for=\"texto\">Introduce una frase:</label><br>
        <textarea name=\"texto\" id=\"texto\" placeholder=\"Escribe aquí...\" required></textarea><br>
        <button class=\"btn\" type=\"submit\">Predecir</button>
    </form>

    {% if resultado is not none %}
    <div class=\"result\">
        <p><strong>Texto ingresado:</strong> {{ texto }}</p>
        <p><strong>Sentimiento predicho:</strong> {{ resultado }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    texto_usuario = ""
    if request.method == 'POST':
        texto_usuario = request.form.get('texto', '')
        resultado = predecir_sentimiento(modelo, texto_usuario)

    return render_template_string(html_template, resultado=resultado, texto=texto_usuario)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

