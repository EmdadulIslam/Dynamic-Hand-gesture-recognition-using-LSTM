import numpy as np

# Provide the actual path to your .npy file
sample_data_path = r"C:\Users\sazza\OneDrive\Documents\newMay31\main_ann\HGD2full\train\antiClockwise\flipRAWantiClockwise1\Res3d_feature.npy"

# Load the data
sample_data = np.load(sample_data_path, allow_pickle=True)

# Print the shape of the data and the type of the first element
print("Shape of the loaded data: ", sample_data.shape)
print("Type of the first element in the loaded data: ", type(sample_data[0]))
