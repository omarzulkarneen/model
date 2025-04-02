from flask import Flask, request, jsonify
import numpy as np
import torch
import joblib

# Define the LSTM model architecture
class EnergyLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnergyLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Initialize the model with the correct architecture
input_size = 5  # Number of input features
hidden_size = 50
num_layers = 2
output_size = 1
model = EnergyLSTM(input_size, hidden_size, num_layers, output_size)

# Load the state dictionary into the model
model.load_state_dict(torch.load("TSCV_model.pth", map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Load the scaler and label encoders
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training
target_scaler = joblib.load("target_scaler.pkl")  # Load the target scaler
label_encoders = {
    "appliance": joblib.load("appliance_encoded.pkl"),
    "occupancy_status": joblib.load("occupancy_status_encoded.pkl")
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.json
        input_data = np.array(data["input_data"])  # Convert list to NumPy array
        
        # Encode categorical variables and prepare input array
        encoded_data = []
        for item in input_data:
            encoded_item = [
                item["temperature_setting_C"],
                label_encoders["occupancy_status"].transform([item["occupancy_status"]])[0],
                item["home_id"],
                label_encoders["appliance"].transform([item["appliance"]])[0],
                item["usage_duration_minutes"]
            ]
            encoded_data.append(encoded_item)

        # Convert the encoded data to a NumPy array
        encoded_data = np.array(encoded_data, dtype=np.float32)
        # Calculate window_size from the input data
        calculated_window_size = encoded_data.shape[0]
        
        # Validate window_size
        if calculated_window_size != 12:
            return jsonify({"error": f"Invalid window_size. Expected 12, but got {calculated_window_size}."}), 400
        
        # Scale the input data
        scaled_input = scaler.transform(encoded_data)
        scaled_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(scaled_input)
        
        # Inverse scale the prediction
        prediction_kWh = target_scaler.inverse_transform(prediction.numpy())
        
        # Format the response
        response = {
            "energy_consumption_6_hours_ahead_kWh": float(prediction_kWh[0][0])
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)