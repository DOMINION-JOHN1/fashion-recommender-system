import pickle
import os
filenames = pickle.load(open('filepath.pkl','rb'))

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

pickle.dump(filenames,open('filepath.pkl','wb'))
