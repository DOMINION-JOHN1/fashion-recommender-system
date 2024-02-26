
import pickle as pickle
import os

filenames = []

# Sort the folder names
folder_names = sorted(['images1', 'images2', 'images3', 'images4', 'images5'])

# Iterate over sorted folder names
for folder_name in folder_names:
    # Iterate over files in each folder
    for file in os.listdir(folder_name):
        # Check if the file is not a shortcut (.lnk)
        if not file.endswith('.lnk'):
            filenames.append(os.path.join(folder_name, file))

pickle.dump(filenames,open('filepath.pkl','wb'))
