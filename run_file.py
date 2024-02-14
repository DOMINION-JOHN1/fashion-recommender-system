import pickle5 as pickle
import os
filenames = pickle.load(open('filenames.pkl','rb'))

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

pickle.dump(filenames,open('filenames.pkl','wb'))
