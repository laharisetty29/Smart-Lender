import joblib
import numpy as np

model = joblib.load("best_model.pkl")
print("Model loaded:", type(model))

# Quick test prediction
sample = np.array([[1, 1, 0, 0, 0, 5000, 1500, 128, 360, 1, 2]])
pred = model.predict(sample)[0]
print("Test prediction:", "Approved" if pred == 1 else "Rejected")
