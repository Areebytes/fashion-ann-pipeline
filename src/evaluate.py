import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = keras.models.load_model("models/model.h5")
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

os.makedirs("reports", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
plt.savefig("reports/confusion_matrix.png", dpi=120)

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
print(f"Test accuracy: {acc:.4f}")