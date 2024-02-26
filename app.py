import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from classification_models.tfkeras import Classifiers

filenames = pickle.load(open('filepath.pkl','rb'))
feature_list = pickle.load(open('feature_list.pkl','rb'))
feature_list = np.array(feature_list)
# Get the ResNeXt model
ResNeXt50, preprocess_input = Classifiers.get('resnext50')
# create a model instance
model = ResNeXt50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
model.trainable = False
model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])

st.title('goody goody fashion store')

def save_uploaded_file(uploaded_file):
    try:
        # Check if the 'uploads' directory exists. If not, create it.
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        # Print the exception message
        print(f"Error in save_uploaded_file: {e}")
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        indices = recommend(features,feature_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        # Add a print statement to print the file path
        print(f"File path: {filenames[indices[0][0]]}")

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")
