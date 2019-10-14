
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input

##from sklearn.metrics import confusion_matrix

print(tf.__version__)

##train = r'D:\Study\COMPX591\Data\singleimages3\train'
#test = r'D:\Study\COMPX591\Data\singleimages\test'

train = '/Scratch/dans/kauri/data/singleimages/train'
test = '/Scratch/dans/kauri/data/singleimages/test'


# create a data generator
datagen = ImageDataGenerator(brightness_range=[0.4,1.0], horizontal_flip=True) #brightness_range=[0.4,1.0] #horizontal_flip=True doesnt help accuracy

# load and iterate training dataset
train_generator = datagen.flow_from_directory(train, class_mode='categorical', batch_size=50, shuffle=True)
# load and iterate validation dataset
##val_it = datagen.flow_from_directory(validate, class_mode='binary', batch_size=64)
# load and iterate test dataset
##test_generator = datagen.flow_from_directory(test, class_mode='categorical', batch_size=100, shuffle=False)

test_generator = datagen.flow_from_directory(test, class_mode='categorical', batch_size=50, shuffle=False)


##train_generator.
##print(train_generator.labels)

# confirm the iterator works
batchX, batchy = train_generator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# Build the model.

IMG_SHAPE = (256,256, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2,activation='sigmoid')

model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  prediction_layer
])

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics = ['accuracy'] ##, precision_threshold(0.00001), recall_threshold(0.5)
  #metrics=['accuracy']
)

model.fit(
  train_generator, # training data
  steps_per_epoch = 340,
  epochs=25,
  batch_size=50
  ##class_weight = { 0 : 1, 1: 5}
)

results = model.evaluate(test_generator)
print('test loss, test acc:', results)


pred = model.predict_generator(test_generator, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
report = classification_report(np.argmax(pred, axis=1), predicted_class_indices, digits = 20)
print(report)
              
print(predicted_class_indices)
##print(predictions)
##print (pred) ##-- probability of each class unnormalised

X_test, y_test = test_generator.next()

incorrects = np.nonzero(predicted_class_indices.reshape((-1,)) != test_generator.labels)
for ind in incorrects[0]:
    print(test_generator.filenames[ind])
    print (pred[ind])



##confusion_matrix(y_true, y_pred)

model.save('model_vgg.h5')
#model.save_weights('model.h5')
#predictions = model.predict(test_generator)

## Print our model's predictions.
#predicted_class_indices = np.argmax(predictions, axis=1) # [7, 2, 1, 0, 4]
#print (predicted_class_indices)
#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]
#print (predictions)
## Check our predictions against the ground truths.

## = model.predict_proba(test_generator)
#for i in range(len(predictions)):
#	print("X=%s, Predicted=%s" % (test_generator[i]))


