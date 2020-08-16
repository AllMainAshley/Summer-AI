import os
import zipfile
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Unzip
# local_zip='rps-test-set.zip'
# zip_ref=zipfile.ZipFile(local_zip,'r')
# zip_ref.extractall()
# zip_ref.close()

# Define Directory
train_dir='rps'
validation_dir='rps-test-set'
train_rock_dir=os.path.join(train_dir,'rock')
train_paper_dir=os.path.join(train_dir,'paper')
train_scissors_dir=os.path.join(train_dir,'scissors')
validation_rock_dir=os.path.join(validation_dir,'rock')
validation_paper_dir=os.path.join(validation_dir,'paper')
validation_scissors_dir=os.path.join(validation_dir,'scissors')

# ImageDataGenenrator
train_datagen= ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)
test_datagen=ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(300,300),
    batch_size=32,
    class_mode='categorical'    # 这不再是二分类所以，class_mode要更改为categorical
)
validation_generator=test_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'  # 这不再是二分类所以，class_mode要更改为categorical
)

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
history=model.fit_generator(
    train_generator,
    steps_per_epoch=28,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=4,
    verbose=2
)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs,loss )
plt.plot(epochs,val_loss)
plt.title('Training and validation loss')
plt.figure()