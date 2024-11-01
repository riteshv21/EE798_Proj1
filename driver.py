from PETS_counting_train_val import build_model_load_Weights, rgb2gray, check_folder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from math import *
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

def optinal_fusion_model():
    build_model_load_Weights()
    label_df = pd.read_csv("coords_correspondence/fusion_dataset_converted_npz/labels.csv")
    label_df.columns = ['id', 'people']

    img = np.load("coords_correspondence/fusion_dataset_converted_npz/images.npy")

    labels = np.array(label_df['people'])

    x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.3)

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(480,640,3), activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), metrics=['mae'])

    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=80, batch_size=8)

    model.save("models/final_trained/trained_model_30_50.h5")
    
def display_results():
    label_df = pd.read_csv('coords_correspondence/fusion_dataset_converted_npz/labels.csv')
    label_df.columns = ['id', 'people']

    img = np.load('coords_correspondence/fusion_dataset_converted_npz/images.npy')
    labels = np.array(label_df['people'])

    x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.3)

    model = load_model('models/final_trained/trained_model_30_50.h5')
    predictions = model.predict(x_test).flatten()
    temp = []
    for prediction in predictions:
        temp.append(int(round(prediction)))
    predictions = np.array(temp)
    mae = np.mean(np.abs(predictions - y_test))
    print("Actual values v/s the predicted values by the model are mentioned in the following table:")
    data = {
        "Actual value": y_test,
        "Predictions": predictions
    }
    df = pd.DataFrame(data)
    print(df)
    print(f"The Mean Absolute Error (MAE) of the model rounded off to two digits comes out to be {mae:.2f}")
    
if __name__ == "__main__":
    optinal_fusion_model()
    display_results()
