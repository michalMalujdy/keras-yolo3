"""
Retrain the YOLO model for your own dataset.
"""
import sys
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model_builder import create_model, create_tiny_model, get_classes, get_anchors
from yolo3.model import preprocess_true_boxes, yolo_loss
from yolo3.utils import get_random_data

train_dir = './train_data'

def _main(optimizer_str = 'Adam', learning_rate = 0.0001, epochs_count = 50, batch_size = 32, log_dir = '', checkpoint_dir = ''):

    printConfig(optimizer_str, learning_rate, epochs_count, batch_size)

    annotation_path = train_dir + '/annotations.txt'
    classes_path = train_dir +'/custom_classes.txt'
    anchors_path = train_dir + '/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path = train_dir + '/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path = train_dir + '/yolo_v3_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir = log_dir)

    checkpointCallback = ModelCheckpoint(
        checkpoint_dir + '/' + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    optimizer = resolve_optimizer(optimizer_str, learning_rate)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer = optimizer, loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs_count,
                initial_epoch=0,
                callbacks=[logging, checkpointCallback])
        model.save_weights(train_dir + '/trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpointCallback, reduce_lr, early_stopping])
        model.save_weights(train_dir + '/trained_weights_final.h5')

    # Further training if needed.


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def resolve_optimizer(optimizer_str, learning_rate):
    if optimizer_str is 'RMSprop':
        return RMSprop(lr = learning_rate)
    else:
        return Adam(lr = learning_rate)

def printConfig(optimizer_str, learning_rate, epochs_count, batch_size):
    print('----------------------------------------------------------')
    print('Set configuration:')
    print('Optimizer: {}'.format(optimizer_str))
    print('Learning rate: {}'.format(learning_rate))
    print('Epochs count: {}'.format(epochs_count))
    print('Batch size: {}'.format(batch_size))
    print('----------------------------------------------------------')

if __name__ == '__main__':
    _main()
