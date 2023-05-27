from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import warnings
import pandas as pd
from tqdm import tqdm
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import sys

sns.set()

from train_1705074 import *

from sklearn.metrics import confusion_matrix, accuracy_score, \
        precision_score, recall_score, f1_score, log_loss, ConfusionMatrixDisplay


with open('parameters_30000_20_001.pkl', 'rb') as f:
    conv1_filters, conv1_bias, \
    conv2_filters, conv2_bias, \
    fc1_filters, fc1_bias, \
    fc2_filters, fc2_bias, \
    fc3_filters, fc3_bias = pickle.load(f)



def load_test_dataset(path_to_folder, output_path):
    
    image_dir = path_to_folder
    images_data = []

    list_dir = sorted(os.listdir(image_dir))

    for i, img_file in enumerate(list_dir):

        if not img_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img = Image.open(os.path.join(image_dir, img_file))

        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.MaxFilter(3))
        img = img.resize((28, 28))

        img = np.array(img)

        images_data.append(img)


    print(len(images_data))

    np.save(f'{output_path}/test_data.npy', images_data)


def load_test_labels(path_to_folder, filename):

    images_data = np.load(f'Predictions/{filename}.npy', allow_pickle=True)
    df = pd.read_csv(f'{path_to_folder}.csv')
    labels = df['digit'].values

    # if there is no csv file, randomly generate labels
    '''
    labels = np.random.randint(0, 10, len(images_data))
    '''

    print(images_data.shape, labels.shape)
    np.save(f'Predictions/{filename}_labels.npy', np.array(list(zip(images_data, labels))))


def get_test_data(filename):
    images = np.load(f'Predictions/{filename}.npy', allow_pickle=True)
    return images


def load_image_filename(path_to_folder):
    df = pd.read_csv(f'{path_to_folder}.csv')
    images_name = df['filename'].values.tolist()

    # if there is no csv file, collect the images name from the given folder directly
    '''
    image_dir = path_to_folder
    images_name = sorted(os.listdir(image_dir))
    '''

    return images_name


def build_model_for_test():

    num_output_units = 10
    num_input_channels = 1

    model = Sequential(batch_size=16, num_epochs=1, verbose=True)
    model.set_parameters(num_input_channels=num_input_channels, num_output_units=num_output_units, 
                image_h=28, image_w=28)
    

    convolution1 = Convolution(num_output_channels=6, kernel_dim=(5, 5), padding=2, stride=1, learning_rate=0.001)
    convolution1.set_parameters(conv1_filters, conv1_bias)
    
    model.add(convolution1)

    model.add(Relu())
    model.add(MaxPooling(kernel_dim=(2, 2), padding=0, stride=2))

    convolution2 = Convolution(num_output_channels=16, kernel_dim=(5, 5), padding=0, stride=1, learning_rate=0.001)
    convolution2.set_parameters(conv2_filters, conv2_bias)

    model.add(convolution2)

    model.add(Relu())
    model.add(MaxPooling(kernel_dim=(2, 2), padding=0, stride=2))

    model.add(Flatten())

    fc1 = FullyConnected(num_output_units=120, learning_rate=0.001)
    fc1.set_parameters(fc1_filters, fc1_bias)
    model.add(fc1)

    fc2 = FullyConnected(num_output_units=84, learning_rate=0.001)
    fc2.set_parameters(fc2_filters, fc2_bias)
    model.add(fc2)

    fc3 = FullyConnected(num_output_units=num_output_units, learning_rate=0.001)
    fc3.set_parameters(fc3_filters, fc3_bias)
    model.add(fc3)

    model.add(Softmax(num_output_units=num_output_units))
    

    return model


def main(path_to_folder):

    load_test_dataset(path_to_folder=path_to_folder, output_path='Predictions')
    load_test_labels(path_to_folder=path_to_folder, filename='test_data')

    images_name = load_image_filename(path_to_folder)

    test_data = get_test_data(filename="test_data_labels")
    print(test_data.shape)
    print(len(images_name))

    model = build_model_for_test()

    input_X, label = model.reshape_format(input=test_data)
    X, y_true = model.reshape_image(input_X, label)
    y_true = np.asarray(y_true, dtype=int)
    y_true = np.argmax(y_true, axis=1)

    y_pred = model.predict(input=test_data)

    cm = confusion_matrix(y_true, y_pred)

    # plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion matrix for {len(y_pred)} test samples (epochs=20, batch_size=16, learning_rate = 0.001)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Predictions/confusion_matrix_001.png')


    pred_df = pd.DataFrame({'FileName': images_name, 'Digit': y_pred})
    pred_df.to_csv('Predictions/1705074_prediction.csv', index=False)



if __name__ == '__main__':

    # sample path - 'Datasets/NumtaDB_with_aug/training-d'
    path_to_folder = sys.argv[1]
    print(path_to_folder)

    main(path_to_folder)
