# inference.py
import torch
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el modelo
class EnhancedDNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Cargar el modelo
input_dim = 10  # Cambia esto si tienes un número diferente de características
model = EnhancedDNN(input_dim).to(device)
model.load_state_dict(torch.load("best_model_all_data.pth", map_location=device))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    inputs = np.array(data["inputs"]).astype(np.float32)
    inputs_tensor = torch.tensor(inputs).to(device)
    with torch.no_grad():
        prediction = model(inputs_tensor).cpu().numpy().tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
