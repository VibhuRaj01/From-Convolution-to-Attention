import tensorflow as tf
from tensorflow.keras import layers
from data import X_train, X_test, y_train, y_test


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

model= tf.keras.Sequential()
model.add(layers.Conv2D(16,(5,5), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32,(5,5), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32,(5,5), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(units=1024 ,activation='relu'))
model.add(layers.Dense(units=128 ,activation='relu'))
model.add(layers.Dense(units=2,activation='softmax'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer='adam',
              metrics=['accuracy'])
model.build(X_train.shape)
hist=model.fit(X_train,y_train, batch_size=16, epochs=2, validation_data=(X_test,y_test))