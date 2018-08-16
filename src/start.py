#!/usr/bin/env python

import gc
import os
import cv2
import json
import keras
import pickle
import random
import logging
import datetime

import numpy as np 
import keras.backend as K 

from collections import defaultdict, namedtuple
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input, Dense, Flatten, subtract, Lambda, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from custom_callbacks import LogEpochStatistics

home = os.path.dirname(os.getcwd())
time_at_start = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

config_file = os.path.join(home, "config", "config_test.json")

with open(config_file, 'r') as f:
    config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

train_dir = config.train.input_dir
valid_dir = config.valid.input_dir
model_dir = os.path.join(home, "models", f"models_{time_at_start}")
tensorboard_dir = os.path.join(home, "analysis", "tensorboard", f"tensorboard_logs_{time_at_start}")
model_type = config.model_details.model_type

log_file = os.path.join(home, "logs", f"log_{time_at_start}.log")
log_format = '[%(asctime)s]\t[%(levelname)s]\t[%(filename)s]\t[%(funcName)s]\t%(message)s'

logging.basicConfig(filename=log_file, level=logging.DEBUG, format=log_format)
status_logger = logging.getLogger("Status")

model_dictionary = {
    "vgg16": {
        "model": VGG16,
        "input_shape": [224, 224, 3]
    },
    "vgg19": {
        "model": VGG19,
        "input_shape": [224, 224, 3]
    },
    "inceptionV3": {
        "model": InceptionV3,
        "input_shape": [299, 299, 3]
    },
    "inception_resnetV2": {
        "model": InceptionResNetV2,
        "input_shape": [299, 299, 3]
    }
}

input_shape = model_dictionary[model_type]["input_shape"]


class ModelArchitectures:
    @staticmethod
    def get_model():
        label_input = Input(input_shape)
        prediction_input = Input(input_shape)

        base_model = model_dictionary[model_type]["model"](include_top=False, weights='imagenet')
        for layer in base_model.layers[:config.model_details.non_trainable_layer_count]:
            layer.trainable = False
        
        encoded_l = base_model(label_input)
        encoded_r = base_model(prediction_input)

        both = subtract([encoded_l, encoded_r])
        both = Lambda(lambda x: abs(x))(both)
        flattened = Flatten()(both)
        flattened = Dropout(config.dropout)(flattened)
        prediction = Dense(1, activation='sigmoid')(flattened)
        siamese_net = Model(inputs=[label_input, prediction_input], outputs=prediction)
        siamese_net.compile(loss="binary_crossentropy", metrics=["accuracy"],
                            optimizer=Adam(lr=config.learning_rate))
        # print(siamese_net.summary())
        return siamese_net


class SiameseLoader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, X_train, batch_size, num_batches):
        # self.X_valid = X_valid
        self.X_train = X_train
        self.num_classes = len(X_train.keys())
        self.width, self.height = input_shape[:2]
        # _, self.num_examples_valid, _, _ = X_valid.shape
        self.batches = list()
        self.create_batches(num_batches=num_batches, batch_size=batch_size)

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

    def create_batches(self, num_batches, batch_size):
        for _ in range(2 * num_batches):
            self.batches.append(self.get_batch(batch_size))

    def yield_batch(self):
        yield random.choice(self.batches)


class Utils:
    @staticmethod
    def load_file(file_name, input_shape):
        if os.path.exists(file_name):
            image = cv2.imread(file_name)
            image.resize(input_shape)
            return image
        print("Error")

    @staticmethod
    def create_dataset(input_folder, identifier, input_shape):
        result = defaultdict(dict)
        for document_type in os.listdir(input_folder):
            print("Loading Document Type: {}".format(document_type))
            document_type_path = os.path.join(input_folder, document_type)
            for document in os.listdir(document_type_path):
                document_path = os.path.join(document_type_path, document)
                result[document_type][document] = Utils.load_file(document_path, input_shape)

        with open(os.path.join(home, "data",
                               f'{identifier}_{input_shape[0]}x{input_shape[1]}x{input_shape[2]}.pickle'), "wb") as f:
            pickle.dump(result, f)

    @staticmethod
    def load_dataset(identifier, input_shape):
        input_file = os.path.join(home, "data",
                                  f"{identifier}_{input_shape[0]}x{input_shape[1]}x{input_shape[2]}.pickle")
        if not os.path.exists(input_file):
            return None
        with open(input_file, "rb") as f:
            result = pickle.load(f)
        return result


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, X_train, batch_size=32, num_batches=32, shuffle=True):
        self.X_train = X_train
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.loader = SiameseLoader(X_train, batch_size=batch_size, num_batches=self.__len__())
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        return next(self.loader.yield_batch())

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        pass


def initialize():
    K.clear_session()
    all_models_folders = [
        os.path.join(home, "models", directory)
        for directory in os.listdir(os.path.join(home, "models"))
        if os.path.isdir(os.path.join(home, "models", directory))
    ]

    if len(all_models_folders) == 0:
        return None
    
    last_train_models_folder = max(all_models_folders, key=os.path.getmtime)
    time_at_start = last_train_models_folder.split('_')[-1]

    model = keras.models.load_model(os.path.join(home, "models",
                                                 f"models_{time_at_start}", f"model_{time_at_start}.h5"))

    return model


if config.mode == "training":
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    model = ModelArchitectures.get_model()

    train_data = Utils.load_dataset("train", input_shape)
    if train_data is None:
        Utils.create_dataset(train_dir, "train", input_shape)
        train_data = Utils.load_dataset("train", input_shape)

    valid_data = Utils.load_dataset("valid", input_shape)
    if valid_data is None:
        Utils.create_dataset(valid_dir, "valid", input_shape)
        valid_data = Utils.load_dataset("valid", input_shape)

    train_generator = DataGenerator(train_data, 
                                    batch_size=config.train.batch_size, 
                                    num_batches=config.train.num_batches)
    valid_generator = DataGenerator(valid_data, 
                                    batch_size=config.valid.batch_size, 
                                    num_batches=config.valid.num_batches)

    del train_data, valid_data

    gc.collect(2)

    callbacks = [
        EarlyStopping(patience=config.patience, monitor='val_loss', verbose=1),
        ModelCheckpoint(os.path.join(model_dir, f"model_{time_at_start}.h5"), 
                        save_best_only=True, period=1),
        TensorBoard(log_dir=tensorboard_dir, write_graph=True, write_grads=None, write_images=False),
        LogEpochStatistics(status_logger)
    ]
    model.fit_generator(generator=train_generator, 
                        validation_data=valid_generator, 
                        epochs=config.num_epochs,
                        callbacks=callbacks)

elif config.mode == "prediction":
    model = initialize()
    test_label_file = os.path.join(config.valid.input_dir,
                                   "0", "imagesb_b_f_b_bfb88e00_2026193480.tif") 
    test_prediction_file_same = os.path.join(config.valid.input_dir,
                                             "0", "imagesb_b_q_i_bqi55a00_505365510+-5511.tif")
    test_prediction_file_different = os.path.join(config.valid.input_dir,
                                                  "1", "imagese_e_f_r_efr50e00_93212869.tif")

    print(model.predict([np.array([Utils.load_file(test_label_file, input_shape)]),
                         np.array([Utils.load_file(test_prediction_file_same, input_shape)])]))

    print(model.predict([np.array([Utils.load_file(test_label_file, input_shape)]),
                        np.array([Utils.load_file(test_prediction_file_different, input_shape)])]))

