import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app


input_data, labels = load_galaxy_data()

print(input_data.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.20, shuffle=True, random_state =  222, stratify=labels)

img_data = ImageDataGenerator(rescale=1./255)
training_iterator = img_data.flow(x_train, y_train, batch_size = 5)

validation_iterator = img_data.flow(x_test, y_test, batch_size = 5)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128,128, 3)))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate = 0.001) , loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

model.fit(training_iterator, steps_per_epoch=len(x_train)/5, epochs=8, validation_data=validation_iterator, validation_steps=len(x_test)/5)

loss, acc = model.evaluate(x_test, y_test, verbose = 0)
print(loss)
print(acc)

from visualize import visualize_activations
visualize_activations(model, validation_iterator)
