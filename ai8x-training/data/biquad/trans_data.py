import numpy as np

# Helper function to convert lines to floats, skipping empty lines
def convert_to_float(line):
    line = line.strip()
    return float(line) if line else None

# Read data from train.x and train.y files
with open("train.x", "r") as file_i:
    data_train = np.array([convert_to_float(line) for line in file_i if convert_to_float(line) is not None], dtype=np.float32)

with open("train.y", "r") as file_o:
    label_train = np.array([convert_to_float(line) for line in file_o if convert_to_float(line) is not None], dtype=np.float32)
    
with open("test.x", "r") as file_i:
    data_test = np.array([convert_to_float(line) for line in file_i if convert_to_float(line) is not None], dtype=np.float32)

with open("test.y", "r") as file_o:
    label_test = np.array([convert_to_float(line) for line in file_o if convert_to_float(line) is not None], dtype=np.float32)

# Reshape data to (-1, 1) shape
data_train = data_train.reshape(-1, 1)
label_train = label_train.reshape(-1, 1)
data_test = data_test.reshape(-1, 1)
label_test = label_test.reshape(-1, 1)

# Save data to .npy files
np.save("data_train.npy", data_train)
np.save("label_train.npy", label_train)
np.save("data_test.npy", data_test)
np.save("label_test.npy", label_test)
