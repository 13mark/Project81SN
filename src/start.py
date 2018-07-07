import os
import cv2
import keras
import pickle

import numpy as np 
import keras.backend as K 

from collections import defaultdict
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, merge, Flatten
from keras.optimizers import Adam
from keras.models import Model

# TODO: Start prediction Module with ImageNet Weights

home = os.path.dirname(os.getcwd())

train_dir = "D://FinalImages//train"
valid_dir = "D://FinalImages//train"
input_shape = (224, 224, 3)


class ModelArchitectures:
    @staticmethod
    def get_model():
        label_input = Input(input_shape)
        prediction_input = Input(input_shape)

        base_model = VGG16(include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        
        encoded_l = base_model(label_input)
        encoded_r = base_model(prediction_input)

        both = merge([encoded_l, encoded_r], mode=lambda x: K.abs(x[0] - x[1]), output_shape=lambda x: x[0])
        flattened = Flatten()(both)
        prediction = Dense(1, activation='sigmoid')(flattened)
        siamese_net = Model(inputs=[label_input, prediction_input], outputs=prediction)
        siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
        print(siamese_net.summary())
        return siamese_net


class SiameseLoader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, X_train):
        # self.X_valid = X_valid
        self.X_train = X_train
        self.num_classes = len(X_train.keys())
        self.width, self.height = input_shape[:2]
        # _, self.num_examples_valid, _, _ = X_valid.shape

    def get_batch(self, n):
        """Create batch of n pairs, half same class, half different class"""
        categories = np.random.choice(self.num_classes, size=(n, ), replace=True)
        pairs = [np.zeros((n, self.height, self.width, 3)) for _ in range(2)]
        targets = np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category_1 = str(categories[i])
            file_1 = np.random.choice(list(self.X_train[category_1].keys()))
            pairs[0][i, :, :, :] = self.X_train[category_1][file_1].reshape(self.width, self.height, 3)
            category_2 = category_1 if i >= n//2 else np.random.choice(list(self.X_train.keys()))
            file_2 = np.random.choice(list(self.X_train[category_2].keys()))
            pairs[1][i, :, :, :] = self.X_train[category_2][file_2].reshape(self.width, self.height, 3)
        return pairs, targets


def load_file(file_name):
    if os.path.exists(file_name):
        image = cv2.imread(file_name)
        image.resize((224, 224, 3))
    else:
        print("Error")
    return image

def create_dataset(input_folder, identifier):
    result = defaultdict(dict)
    for document_type in os.listdir(input_folder):
        print("Loading Document Type: {}".format(document_type))
        document_type_path = os.path.join(input_folder, document_type)
        for document in os.listdir(document_type_path):
            document_path = os.path.join(document_type_path, document)
            result[document_type][document] = load_file(document_path)

    with open(os.path.join(home, "data", f"{identifier}.pickle"), "wb") as f:
        pickle.dump(result, f)


def load_dataset(identifier):
    with open(os.path.join(home, "data", f"{identifier}.pickle"), "rb") as f:
        result = pickle.load(f)
    return result


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, X_train, batch_size=32, shuffle=True):
        """Initialization"""
        self.X_train = X_train
        self.loader = SiameseLoader(X_train)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return 128

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        return self.loader.get_batch(self.batch_size)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        pass


# create_dataset(train_dir, "train")
# create_dataset(valid_dir, "valid")

# train_data = load_dataset("train")
# valid_data = load_dataset("valid")

model = ModelArchitectures.get_model()
print(model.predict([np.array([load_file(r"D:\\FinalImages\Train\0\imagesa_a_m_m_amm07c00_CTRSP-FILES013353-33.tif")]), 
                     np.array([load_file(r"D:\\FinalImages\Train\0\imagesa_a_w_d_awd43f00_0001202299.tif")])]))

print(model.predict([np.array([load_file(r"D:\\FinalImages\Train\0\imagesa_a_m_m_amm07c00_CTRSP-FILES013353-33.tif")]), 
                     np.array([load_file(r"D:\\FinalImages\Train\1\imagesa_a_g_w_agw02a00_1003546746_1003546751.tif")])]))

# train_generator = DataGenerator(train_data, batch_size=32)
# valid_generator = DataGenerator(valid_data, batch_size=256)

# model.fit_generator(generator=train_generator, valid_generator=valid_generator)
