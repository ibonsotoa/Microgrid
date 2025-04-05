from flask import Flask, request, jsonify
from stable_baselines3 import A2C
import inference  # Se asume que inference.py está en el mismo directorio

app = Flask(__name__)

def infer():
    """
    Función que configura el entorno, carga el modelo y realiza una predicción.
    Devuelve el diccionario de control resultante.
    """
    # Genera la microred usando el generador de Microgrid de inference.py
    env_generator = inference.mg(nb_microgrid=5, path="./data/pymgrid_data")
    env_generator.generate_microgrid(verbose=False)
    microgrid = env_generator.microgrids[0]

    # Crea el entorno a partir del microgrid y reinícialo
    env = inference.MicrogridEnv(microgrid)
    obs, _ = env.reset()

    # Carga el modelo previamente entrenado
    model = A2C.load("a2c_microgrid")

    # Realiza la predicción
    action, _ = model.predict(obs)
    
    # Obtén el diccionario de control usando la función definida en inference.py
    control_dict = inference.actions_agent(microgrid, action)
    env.close()

    return control_dict

@app.route('/inference', methods=['POST'])
def run_inference():
    """
    Endpoint que, al recibir una petición POST, ejecuta la función de inferencia.
    """
    try:
        # Si se requieren parámetros de entrada se pueden extraer de request.get_json()
        result = infer()
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ejecuta el servidor en todas las interfaces en el puerto 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
