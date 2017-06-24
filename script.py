import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout, Reshape

np.random.seed(4)

# Parameters
EPOCHS = 20
NUM_CLASS = 10
BATCH_SIZE = 32
TRAIN_VAL_FRAC = 0.7
IH, IW, IC = 28, 28 , 1
BASE_PATH = "../../../../Datasets/MNIST/"

# Model Version
SAVE_NAME = '0' if len(sys.argv) < 2 else sys.argv[1]
LOAD_NAME = '0' if len(sys.argv) < 3 else sys.argv[2]

# Model Definition
adam = Adam(lr=1e-3, decay=1e-3)

model_in = Input([IH * IW * IC])
model_in_reshape = Reshape(target_shape = [IH, IW, IC])(model_in)

# layer - 1
conv1 = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(model_in_reshape)
conv1_act = LeakyReLU(alpha = 0.3)(conv1)
conv1_act_pool = MaxPool2D(strides=(2, 2))(conv1_act)

# layer - 2
conv2 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(conv1_act_pool)
conv2_act = LeakyReLU(alpha = 0.3)(conv2)
conv2_act_pool = MaxPool2D(strides=(2, 2))(conv2_act)

# layer - 3
conv3 = Conv2D(filters = 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(conv2_act_pool)
conv3_act = LeakyReLU(alpha = 0.3)(conv3)

# layer - 4
flatten4 = Flatten()(conv3_act)

# layer - 5
dense5 = Dense(units = 1024, kernel_regularizer = 'l2', bias_regularizer = 'l2')(flatten4)
dense5_act = LeakyReLU(alpha = 0.3)(dense5)
dense5_act_drop = Dropout(rate = 0.5)(dense5_act)

# layer - 6
dense6 = Dense(units = NUM_CLASS, kernel_regularizer = 'l2', bias_regularizer = 'l2')(dense5_act_drop)
dense6_act = Activation(activation = 'softmax')(dense6)
model_out = dense6_act

model = Model(inputs = model_in, outputs = model_out)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())


# Training or Testing
train_or_test = int(input("Training or Testing (0 / 1) : "))

if train_or_test == 0:
	fresh_or_resume = int(input("Fresh or Resume Training (0 / 1) : "))

	if fresh_or_resume == 1:
		# Load Model Weights
		model.load_weights(filepath = os.path.join("Models", LOAD_NAME + '.h5'))
else:
	# Load Model Weights
	model.load_weights(filepath = os.path.join("Models", LOAD_NAME + '.h5'))

if train_or_test == 0:
	try:
		# Read Trainig Data
		data = pd.read_csv(filepath_or_buffer = os.path.join(BASE_PATH, "train.csv")).values

		data_x = 1.0 - (data[:, 1:] / 255.0)
		data_y = to_categorical(data[:, 0], num_classes = NUM_CLASS)

		idx = np.random.choice(np.arange(data_x.shape[0]), data_x.shape[0], replace = False)
		data_x = data_x[idx]
		data_y = data_y[idx]

		num_train = int(TRAIN_VAL_FRAC * data_x.shape[0])
		train_x = data_x[:num_train]
		train_y = data_y[:num_train]

		val_x = data_x[num_train:]
		val_y = data_y[num_train:]

		# Model Training
		model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs = EPOCHS, batch_size = BATCH_SIZE, shuffle = True)

		# Save Model Weights
		model.save_weights(filepath = os.path.join("Models", SAVE_NAME + '.h5'))
	except:
		print("Backup Created.")

		# Save Model Weights
		model.save_weights(filepath = os.path.join("Models", "Backup", "%s.h5" % (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'))))
else:
	# Read Testing Data
	data = pd.read_csv(filepath_or_buffer = os.path.join(BASE_PATH, "test.csv"))

	test_x = data.values / 255.0

	probs = model.predict(test_x, batch_size = BATCH_SIZE, verbose = 1)
	predictions = np.argmax(probs, axis = 1)

	submission = pd.DataFrame({
					"ImageId" : np.arange(1, test_x.shape[0] + 1),
					"Label" : predictions
				})
	submission.to_csv("kaggle.csv", index = False)
