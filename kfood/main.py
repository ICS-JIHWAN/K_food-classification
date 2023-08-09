import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib

def set_global_seed():
    tf.random.set_seed(2)
SEED = 1

def mk_data(root_path):

    dir = glob.glob(os.path.join(root_path, '*'))
    cls = []
    X   = []
    Y   = []
    for d in dir:
        for c in os.listdir(d):
            p = os.path.join(d, c)
            data = sorted(glob.glob(os.path.join(p, '*')))[:-1]

            cls.append(c)
            X += data
            Y += [c for _ in range(len(data))]

    return X, Y, cls

class DataLoader(Sequence):

    def __init__(self, paths, labels, cls, batch_size, shuffle=True):
        super(DataLoader, self).__init__()

        self.batch_size = batch_size
        self.shuffle    = shuffle

        self.path_list = paths
        self.labels    = labels
        self.classes   = cls
        self.n_classes = len(self.classes)

        self.one_hot_encoding(self.classes)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        indices     = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_path  = [self.path_list[i] for i in indices]
        batch_label = [self.labels[i] for i in indices]

        batch_x = []
        batch_y = []
        for i in range(len(batch_path)):
            # if len(batch_path[i]) <= 3:
            #     print('a')
            if '.jpg' == batch_path[i][-4:] or '.JPG' == batch_path[i][-4:]:
                img = plt.imread(batch_path[i])
                if img.shape[-1] != 3:
                    continue
                img = cv2.resize(img, (300, 300))
                cls = np.zeros((self.n_classes))
                cls[self.label_dict[batch_label[i]]] = 1

                batch_x.append(img / 255.0)
                batch_y.append(cls)

            elif '.png' == batch_path[i][-4:]:
                img = cv2.imread(batch_path[i], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[-1] != 3:
                    continue
                img = cv2.resize(img, (300, 300))
                cls = np.zeros((self.n_classes))
                cls[self.label_dict[batch_label[i]]] = 1

                batch_x.append(img / 255.0)
                batch_y.append(cls)

            elif '.jpeg' == batch_path[i][-5:]:
                img = plt.imread(batch_path[i])
                if img.shape[-1] != 3:
                    continue
                img = cv2.resize(img, (300, 300))
                cls = np.zeros((self.n_classes))
                cls[self.label_dict[batch_label[i]]] = 1

                batch_x.append(img / 255.0)
                batch_y.append(cls)

            elif '.gif' == batch_path[i][-4:]:
                img = plt.imread(batch_path[i])
                if img.shape[-1] != 3:
                    continue
                img = cv2.resize(img, (300, 300))
                cls = np.zeros((self.n_classes))
                cls[self.label_dict[batch_label[i]]] = 1

                batch_x.append(img / 255.0)
                batch_y.append(cls)
            else:
                continue

        return np.array(batch_x), np.array(batch_y)

    def one_hot_encoding(self, classes):
        self.label_dict = dict()
        self.label_inverse_dict = dict()
        for i, c in enumerate(classes):
            self.label_dict[c] = i
            self.label_inverse_dict[i] = c

    def on_epoch_end(self):
        self.indices = np.arange(len(self.path_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

def Modle(layer_dims, initialization='random', num_classes=13):
    set_global_seed()

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(300, 300, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    if initialization == 'he':
        initializer = init_he

    for i in range(len(layer_dims)):
        model.add(Conv2D(layer_dims[i], (3, 3), activation='relu', kernel_initializer=initializer))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='relu'))

    return model

class init_he(tf.keras.initializers.Initializer):
    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stedev = stddev

    def __call__(self, shape, dtype=None):
        return tf.random.normal(
            shape, mean=self.mean, stddev=self.stedev, seed=SEED, dtype=dtype
        )*np.sqrt(2/shape[0])

def step_decay(epoch):
    start       = 0.1
    drop        = 0.5
    epochs_drop = 5.0
    lr          = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

root_path = '/storage/jhchoi/kfood'

X, Y, categories = mk_data(root_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=121)

train_loader = DataLoader(X_train, Y_train, categories, batch_size=32, shuffle=True)
test_loader  = DataLoader(X_test,  Y_test,  categories, batch_size=32, shuffle=True)

num_classes = len(categories)

layer_dims = [64, 128, 128, 256, 256, 512, 512]
model = Modle(layer_dims, initialization='he', num_classes=num_classes)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# lr_scheduler = LearningRateScheduler(step_decay, verbose=1)

with tf.device('/device:GPU:1'):
    history = model.fit(train_loader,
                        validation_data=test_loader,
                        batch_size=32,
                        epochs=30,
                        workers=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 모델 저장
model.save('cats_and_dogs.h5')

# 모델 load
model = tf.keras.models.load_model('kfood.h5')

# 모델 검증
print('testing step:')
score = model.evaluate(test_loader)  # verbose=2
print('loss = ', score[0])
print('accuracy = ', score[1])

# 모델 예측
print('predicting step:')
predict_result = model.predict(X_test)
print('result = ', predict_result)



