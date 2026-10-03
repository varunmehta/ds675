from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 1
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 128, 128

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

f = open("flower/flower_images/flower_labels.csv")
names = f.readlines()

x_train = np.empty((0,128,128))
y_train = np.empty((0,1))
x_test = np.empty((0,128,128))
y_test = np.empty((0,1))

counter = []
for i in range(0, 10):
	counter.append(0)

for i in range(1, len(names), 1):
	names[i] = names[i].strip('\n')
	l = names[i].split(',')
	lab = int(l[1])
	counter[lab] += 1
	if(counter[lab] < 16):
		y_train = np.append(y_train, np.array([[int(l[1])]]), axis=0)
		img = Image.open("flower/flower_images/"+l[0])
		img = img.convert('L')
		img_array = np.array(img)
		x_train = np.append(x_train, np.array([img_array]), axis=0)
	else:
		y_test = np.append(y_test, np.array([[int(l[1])]]), axis=0)
		img = Image.open("flower/flower_images/"+l[0])
		img = img.convert('L')
		img_array = np.array(img)
		x_test = np.append(x_test, np.array([img_array]), axis=0)

print(x_train.shape)
print(x_test.shape)

#exit()

if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, 
optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
