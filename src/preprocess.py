import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_test = np.load("data/raw/y_test.npy")

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y_train,
)

os.makedirs("data/processed", exist_ok=True)
for name, arr in [("x_train", x_tr), ("y_train", y_tr), ("x_val", x_val),
                  ("y_val", y_val), ("x_test", x_test), ("y_test", y_test)]:
    np.save(f"data/processed/{name}.npy", arr)
print("Processed:", x_tr.shape, x_val.shape, x_test.shape)