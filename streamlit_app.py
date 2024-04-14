import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Load the models
models_path = {
    'CNN': 'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\cnn_keras.h5',
    'ResNet50': 'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\resnet_keras.h5',
    'VGG16': 'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\vgg16_keras.h5'
}

models = {name: load_model(path) for name, path in models_path.items()}

# Define the genres
classes = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
           'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport',
           'Thriller', 'War', 'Western']

# Function to predict top 3 genres for a given image and model
def predict_top3_genre(image_path, model):
    img = image.load_img(image_path, target_size=(350, 350, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img)[0]
    top3_indices = np.argsort(y_pred)[::-1][:3]
    top3_genres = [classes[i] for i in top3_indices]
    return top3_genres

# Function to load specific plots from local system
def load_plots(plot_paths):
    plots = [plt.imread(plot_path) for plot_path in plot_paths]
    return plots

# Streamlit app
def main():
    st.sidebar.title('20XD88 - Data Mining Lab')
    st.sidebar.write('Movie Genre prediction from Posters')
    page = st.sidebar.radio("Choose a page", ['Upload Image', 'Display Plots'])

    if page == 'Upload Image':
        st.title('Movie Genre Predictor')
        uploaded_file = st.file_uploader("Upload an image.", type="jpg")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            new_size = (300, 300)  
            img_resized = img.resize(new_size)
            st.image(img_resized, caption='Uploaded Image', use_column_width=False)
            st.write("Predicted genres from each model:")

            results = {}
            for model_name, model in models.items():
                top3_genres = predict_top3_genre(uploaded_file, model)
                results[model_name] = ', '.join(top3_genres)

            df = pd.DataFrame(results.items(), columns=['Model', 'Predicted Genres'])
            st.table(df)


    elif page == 'Display Plots':
        st.title('Some Plots..')
        plot_paths = [
            'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\plots\\time.png',
            'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\plots\\model_acc.png',
            'C:\\Users\\Aditya\\OneDrive - CBMM Supply Services and Solutions Pte Ltd\\Desktop\\Data Science\\Data Mining Lab\\final_package\\plots\\resnet_loss.png'
        ]
        plots = load_plots(plot_paths)
        index = st.slider('Select a plot', 0, len(plots)-1)
        st.image(plots[index])

if __name__ == "__main__":
    main()
