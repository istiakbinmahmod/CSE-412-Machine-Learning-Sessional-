from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import warnings
import pandas as pd
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, \
        precision_score, recall_score, f1_score, log_loss


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def get_mini_batch(input_X, label, batch_size):
    for i in range(0, len(input_X), batch_size):
        yield input_X[i: i + batch_size], label[i: i + batch_size]


def get_image_patches(images, kernel_dim, stride):
    batch_size, image_h, image_w = images.shape
    kernel_h, kernel_w = kernel_dim

    for i in range(image_h - kernel_h + 1):
        for j in range(image_w - kernel_w + 1):
            patch = images[:, i * stride : i * stride + kernel_h, j * stride : j * stride + kernel_w]
            yield i, j, patch


def dilate(input):

    batch_size, input_channels, input_h, input_w = input.shape
    input_copy = np.copy(input)

    # insert zeros between rows and columns
    input_copy = np.insert(input_copy, range(1, input_h), 0, axis=2)
    input_copy = np.insert(input_copy, range(1, input_w), 0, axis=3)

    return input_copy

def pad(input, padding_h, padding_w):
    batch_size, input_channels, input_h, input_w = input.shape
    input_copy = np.copy(input)

    input_copy = np.pad(input_copy, ((0,), (0,), (padding_h,), (padding_w,)), mode='constant')
    
    return input_copy


def getWindows(input, output_shape, kernel_dim, stride=1):

    input_copy = np.copy(input)

    batch_size, output_channels, output_h, output_w = output_shape
    batch_size, input_channels, input_h, input_w = input.shape
    kernel_h, kernel_w = kernel_dim
    batch_stride, channel_stride, kern_h_stride, kern_w_stride = input_copy.strides

    return np.lib.stride_tricks.as_strided(
        input_copy, 
        shape=(batch_size, input_channels, output_h, output_w, kernel_h, kernel_w),
        strides=(batch_stride, channel_stride, stride * kern_h_stride, stride * kern_w_stride, 
                kern_h_stride, kern_w_stride)
    )



def gradient_clipping(gradients, overflow=100.0, underflow=1e-15):
    # loop over each element of gradients to check for overflow and underflow
    
    indices = np.where(np.abs(gradients) > overflow)
    gradients[indices] = np.sign(gradients[indices]) * overflow
    zero_indices = np.where(np.abs(gradients) == 0)
    indices = np.where(np.abs(gradients) < underflow)
    gradients[indices] = np.sign(gradients[indices]) * underflow
    gradients[zero_indices] = 1e-3

    gradients = np.nan_to_num(gradients)

    return gradients


class Convolution:

    def __init__(self, num_output_channels, kernel_dim, padding, stride, learning_rate):

        self.num_input_channels = None
        self.num_output_channels = num_output_channels
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        # kernel_dim = (filter_height, filter_width)
        kernel_h, kernel_w = kernel_dim

        self.kernels = None
        self.bias = None
        self.patches = None
        self.input_shape = None

    def get_class_name(self):
        return self.__class__.__name__

    def get_learned_parameters(self):
        return self.kernels, self.bias

    def has_parameters(self):
        return True

    def set_parameters(self, kernels, bias):
        self.kernels = kernels
        self.bias = bias

    def forward(self, input):

        batch_size, input_channels, image_h, image_w = input.shape
        kernel_h, kernel_w = self.kernel_dim

        self.num_input_channels = input_channels

        # initialize kernels and bias
        if self.kernels is None:
            self.kernels = np.random.randn(self.num_output_channels, self.num_input_channels, 
                                kernel_h, kernel_w) * np.sqrt(2 / (self.num_input_channels * kernel_h * kernel_w))
        
        if self.bias is None:
            # self.bias = np.zeros(self.num_output_channels)
            self.bias = np.random.randn(self.num_output_channels)
        

        output_h = (image_h - kernel_h + 2 * self.padding) // self.stride + 1
        output_w = (image_w - kernel_w + 2 * self.padding) // self.stride + 1

        output = np.zeros((batch_size, self.num_output_channels, output_h, output_w))

        self.input_shape = input.shape
        input_copy = input

        if self.padding != 0:
            input_copy = np.copy(input)
            input_copy = pad(input_copy, self.padding, self.padding)

        self.patches = getWindows(input=input_copy, output_shape=output.shape, kernel_dim=self.kernel_dim,
                            stride=self.stride)

        # vectorize the image and kernel using np.einsum
        output = np.einsum('bihwkl,oikl->bohw', self.patches, self.kernels)
        # output += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        output += self.bias[None, :, None, None]

        return output
    

    def backward(self, grad_prev_layer):
        padding_h = self.padding
        padding_w = self.padding

        kernel_h, kernel_w = self.kernel_dim

        if(self.padding == 0):
            padding_h = kernel_h - 1
            padding_w = kernel_w - 1

        # dilate the gradient of the loss with respect to the output
        grad_copy = np.copy(grad_prev_layer)
        grad_copy = dilate(grad_copy)
        grad_copy = pad(grad_copy, padding_h, padding_w)

        output_patches = getWindows(input=grad_copy, output_shape=self.input_shape, 
                            kernel_dim=self.kernel_dim, stride=1)

        # rotate the kernel 180 degrees around its height and width axes 
        rotated_kernels = np.rot90(self.kernels, 2, axes=(2, 3))

        # calculate the gradient of the loss with respect to the kernels
        grad_kernel = np.einsum('bihwkl,bohw->oikl', self.patches, grad_prev_layer)
        grad_bias = np.sum(grad_prev_layer, axis=(0, 2, 3))
        grad_input = np.einsum('bohwkl,oikl->bihw', output_patches, rotated_kernels)


        # clip the gradients
        grad_input = gradient_clipping(grad_input)

        # update kernels, bias

        self.kernels -= self.learning_rate * grad_kernel
        self.bias -= self.learning_rate * grad_bias

        return grad_input




class Relu:

    def __init__(self):
        self.input = None

    def get_learned_parameters(self):
        return None, None

    def has_parameters(self):
        return False

    def get_class_name(self):
        return self.__class__.__name__
    
    def forward(self, input):
        '''
        self.input = input
        output = np.maximum(0, input)
        return output
        
        '''
        self.mask = (input <= 0)
        output = input.copy()
        output[self.mask] = 0

        return output
    

    def backward(self, grad_output):
        '''
        relu_grad = self.input > 0.0
        return grad_output * relu_grad

        '''
        grad_output[self.mask] = 0
        return grad_output



class MaxPooling:

    def __init__(self, kernel_dim, padding, stride):
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.padding = padding
        self.max_indices = None
        self.output_shape = None
        self.input_shape = None

    def get_learned_parameters(self):
        return None, None
    
    def has_parameters(self):
        return False

    def get_class_name(self):
        return self.__class__.__name__
    
    def forward(self, input):
        # input.shape = (batch_size, num_input_channels, image_h, image_w)
        # output.shape = (batch_size, num_input_channels,
        #                   (image_h - kernel_h + 2 * padding) / stride + 1, 
        #                   (image_w - kernel_w + 2 * padding) / stride + 1))

        batch_size, num_input_channels, image_h, image_w = input.shape
        kernel_h, kernel_w = self.kernel_dim

        output_h = (image_h - kernel_h + 2 * self.padding) // self.stride + 1
        output_w = (image_w - kernel_w + 2 * self.padding) // self.stride + 1

        # initialize output
        output = np.zeros((batch_size, num_input_channels, output_h, output_w))

        self.output_shape = output.shape
        self.input_shape = input.shape

        input_copy = input

        if self.padding != 0:
            input_copy = np.copy(input)
            input_copy = pad(input_copy, self.padding, self.padding)

        self.patches = getWindows(input=input_copy, output_shape=output.shape, kernel_dim=self.kernel_dim,
                            stride=self.stride)
        
        output = np.max(self.patches, axis=(4, 5))

        batch_size, num_input_channels, output_h, output_w = output.shape
        self.max_indices = np.zeros(output.shape).astype(int)
        
        for k in range(batch_size):
            for l in range(num_input_channels):
                for i in range(output_h):
                    for j in range(output_w):
                        self.max_indices[k, l, i, j] = np.argmax(input_copy[k, l, i*self.stride : i*self.stride + kernel_h, 
                                                            j*self.stride : j*self.stride + kernel_w])

        return output

    

    def backward(self, grad_prev_layer):

        # grad_prev_layer.shape = (batch_size, num_input_channels, maxpool_output.shape)

        output = np.zeros(self.input_shape)

        batch_size, num_input_channels, output_h, output_w = self.output_shape
        kernel_h, kernel_w = self.kernel_dim

        for batch in range(batch_size):
            for chnl in range(num_input_channels):
                for hght in range(output_h):
                    for wdth in range(output_w):

                        if(hght * self.stride + kernel_h <= self.input_shape[2] and wdth * self.stride + kernel_w <= self.input_shape[3]):
                            output[batch, chnl, hght * self.stride: hght * self.stride + kernel_h, wdth * self.stride: wdth * self.stride + kernel_w] = np.zeros((kernel_h, kernel_w))
                            output[batch, chnl, hght * self.stride: hght * self.stride + kernel_h, wdth * self.stride: wdth * self.stride + kernel_w].flat[int(self.max_indices[batch, chnl, hght, wdth])] = grad_prev_layer[batch, chnl, hght, wdth]                      


        return output



class Flatten:
    
    def __init__(self):
        self.input_shape = None

    def has_parameters(self):
        return False

    def get_learned_parameters(self):
        return None, None

    def get_class_name(self):
        return self.__class__.__name__
    
    def forward(self, input):
        # input.shape = (batch_size, input_height, input_width, num_input_channels)
        # output.shape = (batch_size, image_h * image_w)

        self.input_shape = input.shape
        output = input.reshape(input.shape[0], -1)

        return output
        
    
    def backward(self, grad_prev_layer):
        output = grad_prev_layer.reshape(self.input_shape)
        return output




class FullyConnected:

    def __init__(self, num_output_units, learning_rate):
        self.num_output_units = num_output_units
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.input = None
        self.has_params = True

    def has_parameters(self):
        return True
    
    def set_parameters(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_class_name(self):
        return self.__class__.__name__

    def get_learned_parameters(self):
        return self.weights, self.bias
    
    def forward(self, input):
        # input.shape = (batch_size, image_h * image_w)
        # output.shape = (batch_size, 10)

        self.input = input

        # initialize weights and bias
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.num_output_units)* np.sqrt(2.0 / input.shape[1])
        
        if self.bias is None:
            self.bias = np.random.randn(self.num_output_units)
            # self.bias = np.zeros(self.num_output_units)

        output = np.dot(input, self.weights) + self.bias
        return output
    

    def backward(self, grad_prev_layer):
        # grad_prev_layer.shape = (batch_size, 10)

        # calculate dW, db, and dX
        dW = np.dot(self.input.T, grad_prev_layer)
        db = np.sum(grad_prev_layer, axis=0)
        dX = np.dot(grad_prev_layer, self.weights.T)

        # clip gradients
        dX = gradient_clipping(dX)

        # update weights and bias
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db

        return dX
        


def cross_entropy_loss(y_hat):
    # y.shape = (batch_size, num_output_units, 1)
    # y_hat.shape = (batch_size, num_output_units, 1)
    # output.shape = (batch_size, num_output_units, 1)

    # calculate output
    # output = -y * np.log(y_hat)
    output = -np.log(y_hat)
    return output
            
def cross_entropy_loss_derivative(y_hat):
    # y.shape = (batch_size, num_output_units, 1)
    # y_hat.shape = (batch_size, num_output_units, 1)
    # output.shape = (batch_size, num_output_units, 1)

    # calculate output
    # output = -y / y_hat
    output = -1 / y_hat
    return output


class Softmax:

    def __init__(self, num_output_units):
        self.num_output_units = num_output_units
        self.input = None

    def get_learned_parameters(self):
        return None, None
    
    def has_parameters(self):
        return False

    def get_class_name(self):
        return self.__class__.__name__

    def get_probability(self, input):
        # input.shape = (batch_size, num_output_units, 1)
        # output.shape = (batch_size, num_output_units, 1)

        # calculate output
        # avoid overflow while calculating exp
        input = input -  np.max(input, axis=1, keepdims=True)
        exp = np.exp(input)
        output = exp / np.sum(exp, axis=1, keepdims=True)
        return output    
    
    
    def forward(self, input):
        # input.shape = (batch_size, num_output_units, 1)
        # output.shape = (batch_size, num_output_units)

        # calculate output
        self.input = input
        output = self.get_probability(input)
        return output
    

    def backward(self, grad_prev_layer):
        output = grad_prev_layer
        return output



class Sequential:
    
    def __init__(self, batch_size=16, num_epochs=10, verbose=True):
        self.layers = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
    
    def add(self, layer):
        self.layers.append(layer)

    def get_layers(self):
        return self.layers

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_parameters(self, num_input_channels, num_output_units, image_h, image_w):
        self.num_input_channels = num_input_channels
        self.num_output_units = num_output_units
        self.image_h = image_h
        self.image_w = image_w


    def reshape_format(self, input):
        # input = (1200, 2)
        # input[0] = (128, 128)
        # input[1] = (1,)

        # X = (1200, 128, 128)
        # y = (1200, 1)

        X = np.array([x[0] for x in input])
        y = np.array([x[1] for x in input])

        print('in reshape_format', X.shape, y.shape)

        return X, y

    
    # this function is used for toy dataset
    def reshape(self, input_X, label):
        
        # input.shape = (500, 4)
        X_full = input_X
        y_full = label

        target_shape = (self.batch_size, self.num_input_channels, self.image_h, self.image_w)

        X_full = X_full.reshape(target_shape)
        y_full = np.eye(self.num_output_units)[y_full.astype(int)]

        # print('in reshape', X_full.shape, y_full.shape)

        return X_full, y_full

    
    # this function is used for image dataset
    def reshape_image(self, input_X, label):
        # input_X shape = (1200, 128, 128)
        # label shape = (1200,)

        # X shape = (1200, 1, 128, 128)
        # y shape = (1200, 1)

        X_full = input_X.reshape(input_X.shape[0], 1, input_X.shape[1], input_X.shape[2])
        # y = label.reshape(label.shape[0], 1)

        y_full = np.eye(self.num_output_units)[label.astype(int)]

        # print('in reshape_image', X_full.shape, y_full.shape)

        return X_full, y_full



    def train(self, input_X, label):

        accuracy = []
        logLoss = []

        for X_batch, y_batch in tqdm(get_mini_batch(input_X=input_X, label=label, batch_size=self.batch_size), desc="Training"):

            # X, y_true = self.reshape(X_batch, y_batch)
            X, y_true = self.reshape_image(X_batch, y_batch)

            output = X

            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)
                # print(f'{layer.get_class_name()} output= {output}')

                # check if any nan value in output
                if np.isnan(output).any():
                    print('nan value in output')
                    print(f'{layer.get_class_name()} {output}')

            y_pred = output

            loss = y_pred - y_true
            grad_output = loss

            logLoss.append(log_loss(y_true, y_pred))

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.asarray(y_true, dtype=int)
            y_true = np.argmax(y_true, axis=1)

            # print(f'y_true = {y_true}, y_pred = {y_pred}')

            # calculate training accuracy, loss
            accuracy.append(accuracy_score(y_true, y_pred))

            # create class labels for log_loss
            # y_true = np.eye(self.num_output_units)[y_true.astype(int)]

            # backward propagation
            for layer in reversed(self.layers):
                grad_output = layer.backward(grad_output)
                # print(f'{layer.get_class_name()} grad_output= {grad_output}')

                # check if any nan value in grad_output
                if np.isnan(grad_output).any():
                    print('nan value in grad_output')
                    print(f'{layer.get_class_name()} {grad_output}')

            
        # calculate training accuracy and loss
        print('Training accuracy: ', np.mean(accuracy))
        print('Training loss: ', np.mean(logLoss))
        

    def normalize(self, X):
        # apply standard normalization on x. mean = 0, std = 1
        X = (X - np.mean(X)) / np.std(X)

        return X

        
    def fit(self, input, validation):

        '''
        X, y = input[:, :-1], input[:, -1] - 1
        '''

        # X, y = input[:, :-1], input[:, -1] - 1
        X, y = self.reshape_format(input)
        X = self.normalize(X)

        for epoch in range(self.num_epochs):
            self.train(input_X=X, label=y)

            if self.verbose:
                print(f"\nEpoch:  {epoch+1} \n")

            # after each epoch, run validation
            self.evaluate(input=validation)


    def get_learned_parameters(self):

        all_parameters = []

        for layer in self.layers:
            param1, param2 = layer.get_learned_parameters()

            if param1 is not None:
                all_parameters.append(param1)
            if param2 is not None:
                all_parameters.append(param2)
        
        return all_parameters


    def predict(self, input):

        '''
        input_X, label = input[:, :-1], input[:, -1] - 1
        '''

        # input_X, label = input[:, :-1], input[:, -1] - 1
        input_X, label = self.reshape_format(input)
        input_X = self.normalize(input_X)

        cnt = 0
        y_pred_list = []

        self.accuracy = []
        self.f1 = []

        for X_batch, y_batch in tqdm(get_mini_batch(input_X=input_X, label=label, batch_size=self.batch_size), desc="Predicting"):
            
            # X, y_true = self.reshape(X_batch, y_batch)
            X, y_true = self.reshape_image(X_batch, y_batch)
            
            output = X

            for layer in self.layers:
                output = layer.forward(output)
                # print(f'{layer.get_class_name()} {output}')

            y_pred = output
            # print(f'output {np.argmax(output)}')

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.asarray(y_true, dtype=int)
            y_true = np.argmax(y_true, axis=1)

            y_pred_list.extend(y_pred)

            # print(y_true, y_pred)

            cnt += np.sum(y_true == y_pred)

            self.accuracy.append(accuracy_score(y_true, y_pred))
            self.f1.append(f1_score(y_true, y_pred, average='macro'))

            # if np.argmax(y_pred) == np.argmax(y_true):
            #     cnt += 1
            # else:
            #     print(y_true, y_pred)


        # print('cnt:', cnt)
        # print("Accuracy: ", cnt / len(input) * 100, "%")

        self.accuracy = np.mean(self.accuracy)
        self.f1 = np.mean(self.f1)

        print("Test Accuracy: ", self.accuracy)
        print("Test F1 Score: ", self.f1)

        return y_pred_list
        
    
    def evaluate(self, input):

        input_X, label = self.reshape_format(input)
        input_X = self.normalize(input_X)

        self.accuracy = []
        self.f1 = []
        self.loss = []

        cnt = 0

        for X_batch, y_batch in tqdm(get_mini_batch(input_X=input_X, label=label, batch_size=self.batch_size), desc="Predicting"):
            
            X, y_true = self.reshape_image(X_batch, y_batch)
            
            output = X

            for layer in self.layers:
                output = layer.forward(output)

            y_pred = output
            # print(f'output {np.argmax(output)}')

            self.loss.append(log_loss(y_true, y_pred))

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.asarray(y_true, dtype=int)
            y_true = np.argmax(y_true, axis=1)

            cnt += np.sum(y_true == y_pred)

        
            # calculate sklearn metrics by batch
            self.accuracy.append(accuracy_score(y_true, y_pred))
            self.f1.append(f1_score(y_true, y_pred, average='macro'))


        self.accuracy = np.mean(self.accuracy)
        self.f1 = np.mean(self.f1)
        self.loss = np.mean(self.loss)

        print("Validation Accuracy: ", self.accuracy)
        print("Validation F1 Score: ", self.f1)
        print("Validation Loss: ", self.loss)
    



def load_dataset(train_data_folder, output_path):
    
    train_data_path = "Datasets/NumtaDB_with_aug"
    image_dir = os.path.join(train_data_path, train_data_folder)
    images_data = []

    list_dir = sorted(os.listdir(image_dir))

    for i, img_file in enumerate(list_dir):

        if not img_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img = Image.open(os.path.join(image_dir, img_file))

        # convert to grayscale
        img = ImageOps.grayscale(img)

        # invert image
        img = ImageOps.invert(img)

        # dilate image
        img = img.filter(ImageFilter.MaxFilter(3))

        # resize to 128x128
        img = img.resize((28, 28))

        img = np.array(img)

        # normalize image
        # img = img / 255.0

        images_data.append(img)


    print(len(images_data))

    np.save(f'{output_path}/{train_data_folder}.npy', images_data)



def load_labels(filename):

    images_data = np.load(f'Nonnormalized/{filename}.npy', allow_pickle=True)
    df = pd.read_csv(f'Datasets/NumtaDB_with_aug/{filename}.csv')

    labels = df['digit'].values

    print(images_data.shape, labels.shape)
    np.save(f'Nonnormalized/{filename}_labels.npy', np.array(list(zip(images_data, labels))))



def build_image_from_array(img_arr):
    image = Image.fromarray(img_arr)
    image.show()



def train_val_split(train_size=0.8, shuffle=True):
    
    # load npy files
    images_A = np.load('Nonnormalized/training-a_labels.npy', allow_pickle=True)
    images_B = np.load('Nonnormalized/training-b_labels.npy', allow_pickle=True)
    images_C = np.load('Nonnormalized/training-c_labels.npy', allow_pickle=True)

    # print(images_A.shape, images_B.shape, images_C.shape)

    images = list(images_A)

    for image in images_B:
        images.append(image)
    
    for image in images_C:
        images.append(image)
    

    # convert images to numpy array
    images = np.array(images)
    # print(images.shape)

    # shuffle the images
    if shuffle:
        np.random.shuffle(images)
    
    
    # take first 2000 images
    # images = images[:20000]


    # split the data into train and validation sets
    train_size = int(train_size * len(images))
    train_data = images[:train_size]
    val_data = images[train_size:]

    return train_data, val_data



def build_model():

    num_output_units = 10
    num_input_channels = 1

    model = Sequential(batch_size=16, num_epochs=20, verbose=True)
    model.set_parameters(num_input_channels=num_input_channels, num_output_units=num_output_units, 
                image_h=28, image_w=28)
    
    model.add(
        Convolution(num_output_channels=6, kernel_dim=(5, 5), padding=2, stride=1, learning_rate=0.001)
    )

    model.add(Relu())
    model.add(
        MaxPooling(kernel_dim=(2, 2), padding=0, stride=2)
    )

    model.add(
        Convolution(num_output_channels=16, kernel_dim=(5, 5), padding=0, stride=1, learning_rate=0.001)
    )

    model.add(Relu())
    model.add(
        MaxPooling(kernel_dim=(2, 2), padding=0, stride=2)
    )

    model.add(Flatten())

    model.add(FullyConnected(num_output_units=120, learning_rate=0.001))
    model.add(FullyConnected(num_output_units=84, learning_rate=0.001))
    model.add(FullyConnected(num_output_units=num_output_units, learning_rate=0.001))

    model.add(Softmax(num_output_units=num_output_units))
    

    return model


def save_parameters_to_pickle(parameters, filename):
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)



def main():

    # load_dataset(train_data_folder="training-a", output_path="Nonnormalized")
    # load_dataset(train_data_folder="training-b", output_path="Nonnormalized")
    # load_dataset(train_data_folder="training-c", output_path="Nonnormalized")
    # load_dataset(train_data_folder="training-d", output_path="Nonnormalized")
    # load_dataset(train_data_folder="training-e", output_path="Nonnormalized")

    # load_labels(filename="training-a")
    # load_labels(filename="training-b")
    # load_labels(filename="training-c")
    # load_labels(filename="training-d")
    # load_labels(filename="training-e")

    # images_B = np.load('Nonnormalized/training-b_labels.npy', allow_pickle=True)
    # print(images_B.shape)

    # for i in range(0, 5):
    #     print(images_B[i][0].shape, images_B[i][1])

    # build_image_from_array(images_B[13][0])

    # '''
    train_data, val_data = train_val_split(train_size=0.9, shuffle=True)
    print(train_data.shape, val_data.shape)


    '''
    df = pd.read_csv("Datasets/Toy_Dataset/trainNN.txt", delimiter=r'\s+', header=None)
    train_data = df.to_numpy()
    print('train_data.shape', train_data.shape)

    df = pd.read_csv("Datasets/Toy_Dataset/testNN.txt", delimiter=r'\s+', header=None)
    test_data = df.to_numpy()
    print('test_data.shape', test_data.shape)
    '''
   
    model = build_model()
    model.fit(input=train_data, validation=val_data)

    all_parameters = model.get_learned_parameters()
    # print('conv_kernels: ', conv_kernels)
    # print('conv_bias: ', conv_bias)

    # save parameters to pickle
    save_parameters_to_pickle(all_parameters, filename="parameters_44359_20_001.pkl")



if __name__ == '__main__':
    main()











