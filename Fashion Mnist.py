import tensorflow as tf
import keras
# import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):     # 在迭代结束时调用
        if(logs.get('loss')<0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 打印第0个样本的值
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

training_images = training_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  #每个图片的像素是28x28
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax) #因为有10种衣物，所以应该得到的是他们各自发概论
])

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),loss='sparse_categorical_crossentropy')

model.fit(training_images,training_labels,epochs=5,callbacks=[callbacks])

model.evaluate(test_images,test_labels)



