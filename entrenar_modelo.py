import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import re

def cargar_datos(archivo):
    preguntas = []
    respuestas = []
    with open(archivo, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Elimina el caracter problemático \u200d y otros raros
                line = re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', line)
                pregunta, respuesta = line.strip().split(':')
                preguntas.append(pregunta.strip())
                respuestas.append(respuesta.strip())
            except ValueError:
                continue  # Ignorar líneas mal formateadas
    return np.array(preguntas), np.array(respuestas)

def convertir_etiquetas(respuestas):
    etiquetas = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    return np.array([etiquetas[resp] for resp in respuestas])

def verificar_duplicados_total(textos):
    total = len(textos)
    unicos = len(set(textos))
    if total != unicos:
        print(f"Advertencia: Se encontraron {total - unicos} duplicados en el dataset total.")
    else:
        print("No se encontraron duplicados en el dataset total.")

def verificar_duplicados_train_test(train_texts, test_texts):
    duplicates = set(train_texts).intersection(set(test_texts))
    if duplicates:
        print(f"Advertencia: Se encontraron {len(duplicates)} ejemplos duplicados entre entrenamiento y prueba.")
        print("Algunos ejemplos duplicados:", list(duplicates)[:5])
    else:
        print("No se encontraron duplicados entre entrenamiento y prueba.")

def construir_modelo(text_vectorizer, num_clases=3):
    model = tf.keras.Sequential([
        text_vectorizer,
        tf.keras.layers.Embedding(
            input_dim=len(text_vectorizer.get_vocabulary()) + 1,
            output_dim=32,
            mask_zero=True
        ),
        tf.keras.layers.LSTM(256, dropout=0.3, recurrent_dropout=0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def entrenar_modelo(model, x_train, y_train, x_val, y_val, epochs):
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=True, validation_data=(x_val, y_val))

def evaluar_modelo(model, x_test, y_test):
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['negativo', 'neutro', 'positivo'])
    print(f"Precisión del modelo: {acc:.4f}")
    print("Reporte de clasificación:\n", report)

def validacion_cruzada(preguntas, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    accuracies = []
    for train_index, val_index in skf.split(preguntas, y):
        x_train_fold = preguntas[train_index]
        y_train_fold = y[train_index]
        x_val_fold = preguntas[val_index]
        y_val_fold = y[val_index]
        
        text_vectorizer = tf.keras.layers.TextVectorization(
            output_mode='int',
            max_tokens=500000
        )
        text_vectorizer.adapt(x_train_fold)
        
        model = construir_modelo(text_vectorizer)
        model.fit(x_train_fold, y_train_fold, epochs=15, batch_size=32, verbose=0)
        
        y_pred_probs = model.predict(x_val_fold)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_val_fold, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} - Precisión: {acc:.4f}")
        fold += 1
        
    print(f"Precisión promedio en validación cruzada: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    archivo_datos = "datos_entrenamiento.txt"
    preguntas, respuestas = cargar_datos(archivo_datos)
    
    verificar_duplicados_total(preguntas)
    
    y = convertir_etiquetas(respuestas)
    
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        preguntas, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Conjunto de entrenamiento:")
    print("Distribución de etiquetas:", Counter(y_train))
    print("Conjunto de prueba:")
    print("Distribución de etiquetas:", Counter(y_test))
    
    verificar_duplicados_train_test(x_train_raw, x_test_raw)
    
    # Crear y adaptar la capa de tokenización
    text_vectorizer = tf.keras.layers.TextVectorization(output_mode='int', max_tokens=500000)
    text_vectorizer.adapt(x_train_raw)
    
    # Construir el modelo
    modelo = construir_modelo(text_vectorizer)
    
    # Entrenar el modelo
    entrenar_modelo(modelo, x_train_raw, y_train, x_test_raw, y_test, epochs=15)
    
    # Evaluar el modelo
    evaluar_modelo(modelo, x_test_raw, y_test)
    
    # Guardar el modelo en formato TF nativo
    # Esto creará la carpeta "modelo_sentimiento" con los assets/variables/etc.
    modelo.save("modelo_sentimiento", save_format="tf")
    
    # Si prefieres un único archivo .keras, utiliza:
    # modelo.save("modelo_sentimiento.keras", save_format="keras")
    
    # Guardar el vocabulario del TextVectorization
    vocabulario = text_vectorizer.get_vocabulary()
    
    # Elimina caracteres Unicode problemáticos antes de guardar (opcional)
    vocabulario = [re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', v) for v in vocabulario]
    
    with open("tokenizer_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocabulario, f, ensure_ascii=False)
    
    print("Modelo y tokenizer guardados exitosamente.")
    
    # Validación cruzada
    print("\nRealizando validación cruzada:")
    validacion_cruzada(preguntas, y, n_splits=5)

