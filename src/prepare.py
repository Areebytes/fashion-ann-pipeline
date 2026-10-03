import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

os.makedirs("data/raw", exist_ok=True)
np.save("data/raw/x_train.npy", x_train)
np.save("data/raw/y_train.npy", y_train)
np.save("data/raw/x_test.npy", x_test)
np.save("data/raw/y_test.npy", y_test)
print("Raw data saved:", x_train.shape, x_test.shape)