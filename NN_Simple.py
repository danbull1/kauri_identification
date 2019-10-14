
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.python.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.normalization import BatchNormalization

##from sklearn.metrics import confusion_matrix

print(tf.__version__)

#train = r'D:\Study\COMPX591\Data\singleimages\train'
#test = r'D:\Study\COMPX591\Data\singleimages\test'

train = '/Scratch/dans/kauri/data/singleimages/train'
test = '/Scratch/dans/kauri/data/singleimages/test'

# create a data generator
datagen = ImageDataGenerator(brightness_range=[0.6,1.4],horizontal_flip=True) ##, validation_split=0.2 #brightness_range=[0.4,1.0] #horizontal_flip=True doesnt help accuracy

# load and iterate training dataset
train_generator = datagen.flow_from_directory(train, class_mode='categorical', batch_size=200, shuffle=True)
# load and iterate validation dataset
##val_it = datagen.flow_from_directory(validate, class_mode='binary', batch_size=100)
# load and iterate test dataset
test_generator = datagen.flow_from_directory(test, class_mode='categorical', batch_size=200, shuffle=False)
##x,y = test_generator.class_indices.next()
print(test_generator.filenames)


##train_generator.
##print(train_generator.labels)

# confirm the iterator works
batchX, batchy = train_generator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# Build the model.

model = Sequential()
model.add(Dense(32, activation="relu"))
#model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), strides=(3, 3)))## dim of input shape
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2, 2), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), strides=(2, 2),  padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), strides=(2, 2),  padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), strides=(1, 1),  padding="same", activation="relu"))    
model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding="same", activation="relu")) 
model.add(Flatten())
model.add(Dense(units=2,activation= "sigmoid")) ## a single output
model.add(MBConv)

##https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics = ['accuracy']##, precision_threshold(), recall_threshold()]
  #metrics=['accuracy']
)

model.fit(
  train_generator, # training data
  steps_per_epoch=80,
  epochs=25,
  batch_size=200,
  ##validation_split =0.1,
  ##class_weight = { 0 : 1, 1: 2}
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

model.save('model_simplenn.h5')
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


