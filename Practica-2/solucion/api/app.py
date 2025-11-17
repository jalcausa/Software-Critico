from flask import Flask, request, jsonify
from redis import Redis, RedisError
import os
import socket
import time
import numpy as np
from keras.models import load_model
import json

# --- Conexión a Redis ---
REDIS_HOST = os.getenv('REDIS_HOST', "localhost")
print(f"--- Conectando a Redis en: {REDIS_HOST} ---")

try:
    redis = Redis(host=REDIS_HOST, port=6379, db=0,
                  socket_connect_timeout=2, socket_timeout=2,
                  decode_responses=True)
    redis.ping()
    print("--- Conexión a Redis exitosa ---")
    
    # Crear la serie temporal si no existe
    try:
        redis.execute_command('TS.CREATE', 'mediciones', 'RETENTION', '86400000', 'LABELS', 'sensor', 'datos')
        print("--- Serie temporal 'mediciones' creada ---")
    except:
        print("--- Serie temporal 'mediciones' ya existe ---")
        
except RedisError as e:
    print(f"--- ERROR AL CONECTAR CON REDIS: {e} ---")
    pass

# --- Cargar modelo LSTM y configuración ---
print("--- Cargando modelo LSTM y configuración ---")
try:
    model = load_model('modelo.keras')
    print("--- Modelo LSTM cargado exitosamente ---")
    
    # Cargar configuración (n_steps, k, mae_mean)
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    N_STEPS = config['n_steps']
    K = config['k']
    MAE_MEAN = config['mae_mean']
    UMBRAL = K * MAE_MEAN
    
    print(f"--- Configuración cargada: n_steps={N_STEPS}, k={K}, umbral={UMBRAL:.4f} ---")
    
except Exception as e:
    print(f"--- ERROR AL CARGAR MODELO O CONFIGURACIÓN: {e} ---")
    model = None

app = Flask(__name__)

# --- Endpoint Raíz ---
@app.route("/")
def index():
    """Página principal que muestra ayuda y el hostname."""
    hostname = socket.gethostname()
    html = f"<h3>API de Mediciones</h3>" \
           f"<b>Hostname:</b> {hostname}<br/><br/>" \
           "<b>Endpoints:</b><br/>" \
           "<code>/nuevo?dato=VALOR</code> - Guarda un nuevo valor (ej. 10.5)<br/>" \
           "<code>/listar</code> - Muestra todos los valores guardados"
    return html

# --- Endpoint /nuevo ---
@app.route("/nuevo")
def nuevo():
    """
    Recibe un 'dato' por parámetro query y lo almacena en Redis TimeSeries.
    """
    dato_str = request.args.get("dato")

    if dato_str is None:
        return "Error: No se proporcionó el parámetro 'dato'.<br/>" \
               "Ejemplo de uso: /nuevo?dato=12.3", 400

    try:
        dato_float = float(dato_str)
    except ValueError:
        return f"Error: El valor '{dato_str}' no es un número válido.", 400

    try:
        timestamp = int(time.time() * 1000)  # Timestamp en milisegundos
        redis.execute_command('TS.ADD', 'mediciones', timestamp, dato_float)
        return f"Dato '{dato_float}' guardado correctamente en timestamp {timestamp}.", 200
    except Exception as e:
        hostname = socket.gethostname()
        return f"Error al guardar en Redis desde {hostname}: {e}", 500

# --- Endpoint /listar ---
@app.route("/listar")
def listar():
    """
    Muestra un listado de todas las mediciones almacenadas en Redis TimeSeries.
    """
    hostname = socket.gethostname()
    
    try:
        mediciones = redis.execute_command('TS.RANGE', 'mediciones', '-', '+')
        
        output = f"<h3>Listado de Mediciones</h3>"
        output += f"<b>Servidor (Hostname):</b> {hostname}<br/>"
        output += "<pre>"
        
        if not mediciones:
            output += "No hay mediciones almacenadas."
        else:
            for timestamp, value in mediciones:
                # Convertir timestamp a fecha legible
                fecha = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))
                output += f"{fecha}: {value}\n"
            
        output += "</pre>"
        return output
        
    except Exception as e:
        return f"Error al leer de Redis desde {hostname}: {e}", 500

# --- Endpoint /detectar ---
@app.route("/detectar")
def detectar():
    """
    Recibe una nueva medición, la almacena en Redis y detecta si es una anomalía.
    Usa las últimas N_STEPS mediciones para hacer la predicción.
    
    Ejemplo: /detectar?dato=12.5
    """
    dato_str = request.args.get("dato")
    
    if dato_str is None:
        return jsonify({
            "error": "No se proporcionó el parámetro 'dato'",
            "ejemplo": "/detectar?dato=12.3"
        }), 400
    
    # Validar que el modelo esté cargado
    if model is None:
        return jsonify({
            "error": "El modelo de detección no está disponible"
        }), 500
    
    try:
        dato_float = float(dato_str)
    except ValueError:
        return jsonify({
            "error": f"El valor '{dato_str}' no es un número válido"
        }), 400
    
    try:
        # 1. Almacenar la nueva medición en Redis
        timestamp = int(time.time() * 1000)
        redis.execute_command('TS.ADD', 'mediciones', timestamp, dato_float)
        
        # 2. Obtener las últimas N_STEPS+1 mediciones (necesitamos N_STEPS para predecir y comparar con el nuevo dato)
        mediciones = redis.execute_command('TS.RANGE', 'mediciones', '-', '+')
        
        # Verificar que tengamos suficientes datos
        if len(mediciones) < N_STEPS:
            return jsonify({
                "mensaje": "Medición almacenada. Se necesitan al menos {} mediciones para detectar anomalías".format(N_STEPS),
                "mediciones_actuales": len(mediciones),
                "mediciones_necesarias": N_STEPS,
                "anomalia": "sin_datos_suficientes"
            }), 200
        
        # 3. Extraer las últimas N_STEPS mediciones (ventana) para hacer la predicción
        ultimas_mediciones = mediciones[-N_STEPS:]
        
        # Preparar datos para la respuesta JSON
        ventana_datos = []
        valores_ventana = []
        
        for ts, valor in ultimas_mediciones:
            fecha = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts/1000))
            ventana_datos.append({
                "timestamp": int(ts),
                "fecha": fecha,
                "valor": float(valor)
            })
            valores_ventana.append(float(valor))
        
        # 4. Preparar entrada para el modelo: reshape a [1, n_steps, 1]
        X_input = np.array(valores_ventana).reshape(1, N_STEPS, 1)
        
        # 5. Hacer predicción
        y_pred = model.predict(X_input, verbose=0)[0][0]
        
        # 6. Calcular MAE entre predicción y valor real (el nuevo dato)
        mae = abs(y_pred - dato_float)
        
        # 7. Detectar anomalía
        es_anomalia = mae > UMBRAL
        
        # 8. Preparar respuesta
        response = {
            "dato_recibido": dato_float,
            "prediccion": float(y_pred),
            "mae": float(mae),
            "umbral": float(UMBRAL),
            "anomalia": "si" if es_anomalia else "no",
            "ventana_utilizada": ventana_datos,
            "parametros": {
                "n_steps": N_STEPS,
                "k": K,
                "mae_mean": MAE_MEAN
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        hostname = socket.gethostname()
        return jsonify({
            "error": f"Error en detección desde {hostname}: {str(e)}"
        }), 500

# --- Ejecución Principal ---
if __name__ == "__main__":
    PORT = int(os.getenv('PORT', 5001))
    print(f"--- Iniciando servidor Flask en puerto: {PORT} ---")
    app.run(host='0.0.0.0', port=PORT, debug=True)