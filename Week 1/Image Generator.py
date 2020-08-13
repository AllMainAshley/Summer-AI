from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

# # 实例化图像
train_datagen = ImageDataGenerator(rescale=1./255)      # rescle:数据标准化
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 父文件夹-子文件夹-图片 注意：父文件夹才是我们要选择的文件夹
    target_size=(300,300),    # 调整图片大小使他们一致
    batch_size=128,           # 批量载入图片，比逐个载入效率更高
    class_mode='binary'       # 二分类
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    validatioon_dir,
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')   # 等价于tf.keras.layers.Dense(2,activation='softmax')
])

model.summary()

model.compile(loss='binary_classification',
              optimizer=RMSprop(lr=0.001),
              metircs=['acc'])
# 在分类时尚衣物(fashion mnist)时，loss是分类交叉熵(sparse_categorical_crossentropy)
# 在此处做的是二分类，所以用二值交叉熵(binary_crossentropy)

hitory = model.fit_generator(
    train_generator,     # 从训练目录中流式传输图像，我们一共有1024张图片，每批128个，一共需要8批
    steps_per_epoch=8,   # 因为一共需要8批，所以每次epoch都需要8个steps
    epochs=15,           # 迭代15次
    validation_data=validation_generator,   # 验证集
    validation_steps=8,  # 验证集一共有256张图片，每批32个，一共需要8批
    verbose=2            # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
)


# import numpy as np from google.colab import files # google colab专有代码 from keras.preprocessing import image
# uploaded = files.upload() # google colab专有代码
# for fn in upload.keys(): # google colab专有代码 path = '/content/' + fn
# 循环遍历图片
# img = image.load_imag(path,target_size=(300,300)) x = image.image_to_array(img) x = np.expand_dims(x,axis=0) images = np.vstack(x)
# classes = model.predict(images,bach_size=10) print(classes[0])
