#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import gc
import matplotlib.pyplot as plt
import datetime

#os.environ['CUDA_VISIBLE_DEVICES']='0'
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist

import pickle
from DataAug import *

def load_cifar10(max_class=10,selected_label=None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    if selected_label==None:
        selected_label=list(range(max_class))
    print("label: {} are selected for training".format(selected_label))
    x_train = [x_train[i] for i in range(y_train.shape[0]) if y_train[i] in selected_label]
    x_test = [x_test[i] for i in range(y_test.shape[0]) if y_test[i] in selected_label]

    y_train = [selected_label.index(y_train[i]) for i in range(y_train.shape[0]) if y_train[i] in selected_label]
    y_test = [selected_label.index(y_test[i]) for i in range(y_test.shape[0]) if y_test[i] in selected_label]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    x_mean=np.mean(x_train, axis=0)
    x_train -= x_mean
    x_std=np.std(x_train, axis=0)
    x_train /= x_std

    x_test -= x_mean
    x_test /= x_std

    print("x_train shape: {}, y_train shape: {} ".format(x_train.shape, y_train.shape))
    print("x_test shape: {}, y_test shape: {} ".format(x_test.shape, y_test.shape))
    input_shape = (None, 32, 32, 3)
    return x_train, y_train, x_test, y_test, input_shape

def load_cifar100(max_class=100,selected_label=None):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    if selected_label==None:
        selected_label=list(range(max_class))
    print("label: {} are selected for training".format(selected_label))
    x_train = [x_train[i] for i in range(y_train.shape[0]) if y_train[i] in selected_label]
    x_test = [x_test[i] for i in range(y_test.shape[0]) if y_test[i] in selected_label]

    y_train = [selected_label.index(y_train[i]) for i in range(y_train.shape[0]) if y_train[i] in selected_label]
    y_test = [selected_label.index(y_test[i]) for i in range(y_test.shape[0]) if y_test[i] in selected_label]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    x_mean=np.mean(x_train, axis=0)
    x_train -= x_mean
    x_std=np.std(x_train, axis=0)
    x_train /= x_std

    x_test -= x_mean
    x_test /= x_std

    print("x_train shape: {}, y_train shape: {} ".format(x_train.shape, y_train.shape))
    print("x_test shape: {}, y_test shape: {} ".format(x_test.shape, y_test.shape))
    input_shape = (None, 32, 32, 3)
    return x_train, y_train, x_test, y_test, input_shape
def load_mnist(max_class=10):
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    x_train = x_train[np.where(y_train < max_class)]
    y_train = y_train[np.where(y_train < max_class)]
    x_test = x_test[np.where(y_test < max_class)]
    y_test = y_test[np.where(y_test < max_class)]

    x_mean = np.mean(x_train)
    x_train -= x_mean
    x_test -= x_mean
    x_std = np.std(x_train)
    x_train /= x_std
    x_test /= x_std
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    input_shape = (None, img_rows, img_cols, 1)
    return x_train, y_train, x_test, y_test, input_shape


def compute_weights(model, masks):
    # mask weights
    if isinstance(model,list):
        weights=model.copy()
    else:
        weights = model.get_weights().copy()
    if not masks == None:
        for layer in range(len(weights)):
            weights[layer] = weights[layer] * masks[layer]

    # trainable_list=[]
    # for layer in range(len(model.variables)):
    #     if model.variables[layer].trainable:
    #         trainable_list.append(layer)

    concat_weights = []
    for layer in range(len(weights)):
        if not isinstance(model, list):
            if model.variables[layer].trainable:
                concat_weights.append(tf.reshape(weights[layer], [-1]))
        else:
            concat_weights.append(tf.reshape(weights[layer], [-1]))
    concat_weights = tf.concat(concat_weights, 0).numpy()

    gc.collect()
    return concat_weights


def compute_full_gradient(model, dataset,num_classes=10):
    # loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    num_batches=tf.data.experimental.cardinality(dataset)
    full_gradient = None

    # TRAINING
    train_iter = iter(dataset)

    # iteration of batches
    for batch in range(num_batches):
        train_batch = next(train_iter)

        # create gradientTape, use tape.gradient to computer gradient of certain variable.
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            predictions = model(train_batch[0])
            loss = loss_object(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        if full_gradient == None:
            full_gradient = tf.identity_n(gradients)
        else:
            for layer in range(len(gradients)):
                full_gradient[layer] += gradients[layer]
    for layer in range(len(full_gradient)):
        full_gradient[layer] /= float(num_batches)
    return full_gradient

def compute_jacobian(model,masks,dataset,num_classes=1):
    num_batches = tf.data.experimental.cardinality(dataset)

    # mask weights
    init_weights=model.get_weights().copy()
    weights=model.get_weights()
    for layer in range(len(weights)):
        weights[layer]=weights[layer]*masks[layer]
    model.set_weights(weights)
    
    jacobian=[]
    data_iter=iter(dataset)

    for batch in range(num_batches):
        data_batch=next(data_iter)

        for b in range(data_batch[0].shape[0]):
            gradient_sample=[]
            for c in range(num_classes):
                with tf.GradientTape() as tape:
                    predictions_0=model.logits(data_batch[0][b][np.newaxis,...])[0,c]
                gradients=tape.gradient(predictions_0,model.trainable_variables)
                for layer in range(len(gradients)):
                    gradients[layer]=(tf.reshape(gradients[layer],[-1]))
                gradients=tf.concat(gradients,0).numpy()
                gradient_sample.append(gradients)
            gradient_sample=np.array(gradient_sample)[0,:]
            #print(gradient_sample.shape)
            jacobian.append(gradient_sample)
    jacobian = np.array(jacobian)
    model.set_weights(init_weights)
    gc.collect()
    return jacobian


def compute_jacobian_wrt_true(model, masks, dataset, num_classes=10):
    num_batches = tf.data.experimental.cardinality(dataset)

    # mask weights
    init_weights = model.get_weights().copy()
    weights = model.get_weights()
    for layer in range(len(weights)):
        weights[layer] = weights[layer] * masks[layer]
    model.set_weights(weights)

    jacobian = []
    data_iter = iter(dataset)

    for batch in range(num_batches):
        data_batch = next(data_iter)

        for b in range(data_batch[0].shape[0]):
            c=data_batch[1][b]
            with tf.GradientTape() as tape:
                predictions_0=model.logits(data_batch[0][b][np.newaxis, ...])[0, c]
            gradients = tape.gradient(predictions_0, model.trainable_variables)
            for layer in range(len(gradients)):
                gradients[layer] = (tf.reshape(gradients[layer], [-1]))
            gradients = tf.concat(gradients, 0).numpy()
            jacobian.append(gradients)
    jacobian = np.array(jacobian)
    model.set_weights(init_weights)
    gc.collect()
    return jacobian

class TrainHistory:
    epoch = None
    batch_size=None
    num_classes=None
    train_ds = None
    test_ds = None

    history_weights = []
    init_weights = None
    final_weights = None
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    nonzero = None
    pruning_method = None
    JJ_final=None
    JJ_init=None
    JJ_final_submean=None
    JJ_init_submean=None
    correlation_final=[]
    correlation_init=[]
    correlation_init_submean=[]
    correlation_final_submean=[]
    pruned_acc=[]
    pruned_loss=[]

    def __init__(self):
        self.history_weights = []
        self.correlation_final = []
        self.correlation_init = []
        self.correlation_init_submean = []
        self.correlation_final_submean = []
        self.pruned_acc = []
        self.pruned_loss = []
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        pass

    @classmethod
    def load(self, path):
        with open(path,'rb') as f:
            return pickle.load(f)

    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)

import tensorflow_addons as tfa

def func_aug(x, y):
    width=x.shape[1]
    height=x.shape[2]
    channel=x.shape[3]
    x = tf.image.resize_with_pad(
       x, tf.cast(width*1.2,tf.int32), tf.cast(height*1.2,tf.int32),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.random_crop(x, [tf.shape(x)[0], width, height, channel])
    #x = tf.image.random_flip_left_right(x)
    return x,y

    # tx=tf.random.uniform(shape=[], minval=-4, maxval=4, dtype=tf.int64)
    # ty=tf.random.uniform(shape=[], minval=-4, maxval=4, dtype=tf.int64)
    # x=tfa.image.transform(x,[1,0,-tx,0,1,-ty,0,0],'NEAREST')
    #
    # x = tf.image.random_flip_left_right(x)
    # return x, y

def mask_training(model,masks,train_ds,test_ds,lr=1e-2,epoch=20,num_classes=1,
                  history=None,set_init=0,augmentation=False,opt='SGD',reduceLROnPlateau=False):
    #ori_train_ds=train_ds.copy()
    if history==None:
        history=TrainHistory()
    history.epoch=epoch
    history.train_ds=train_ds
    history.test_ds=test_ds
    history.init_weights = model.get_weights().copy()

    # Training
    EPOCHS = epoch
    num_train_batches=tf.data.experimental.cardinality(train_ds)
    num_test_batches=tf.data.experimental.cardinality(test_ds)

    # mask weights
    if set_init==0:
        weights=model.get_weights()
        if not masks == None:
            for layer in range(len(weights)):
                weights[layer] = weights[layer] * masks[layer]
        model.set_weights(weights)
    
    # loss and optimizer
    if num_classes==1:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy()
    if opt=='SGD':
        optimizer = tf.keras.optimizers.SGD(lr=lr,momentum=0.0, nesterov=False)
    elif opt=='ADAM':
        optimizer = tf.keras.optimizers.Adam(lr=lr)

    if reduceLROnPlateau:
        reducer=ReduceLROnPlateau(
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6
        )
    for epoch in range(EPOCHS):

        history.history_weights.append(model.get_weights().copy())
        gc.collect()

        optimizer.lr = lr
        if epoch>80:
            optimizer.lr=lr*1e-1
        if epoch>120:
            optimizer.lr=lr*1e-2
        if epoch>160:
            optimizer.lr=lr*1e-3
        if epoch>180:
            optimizer.lr=lr*0.5e-3

        # initialize dataset iterators
        train_iter=iter(train_ds)
        test_iter=iter(test_ds)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        if num_classes==1:
            train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        else:
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        # test epoch (ON TEST DATA)
        for batch in range(num_test_batches):
            test_batch=next(test_iter)
            predictions=model(test_batch[0])
            if num_classes==1:
                loss=loss_object(test_batch[1],predictions)
                test_accuracy(test_batch[1], predictions)
            else:
                loss = loss_object(tf.keras.utils.to_categorical(test_batch[1], num_classes=num_classes), predictions)
                test_accuracy(tf.keras.utils.to_categorical(test_batch[1], num_classes=num_classes), predictions)
            test_loss(loss)


        # test epoch (ON TRAIN DATA)
        for batch in range(num_train_batches):
            train_batch=next(train_iter)
            predictions=model(train_batch[0])
            if num_classes==1:
                loss=loss_object(train_batch[1],predictions)
                train_accuracy(train_batch[1], predictions)
            else:
                loss = loss_object(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)
                train_accuracy(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)
            train_loss(loss)


        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        tf.print(template.format(epoch,
                                 np.round(train_loss.result(), 5),
                                 np.round(train_accuracy.result(), 5),
                                 np.round(test_loss.result(), 5),
                                 np.round(test_accuracy.result(), 5)))
        if reduceLROnPlateau:
            optimizer.lr=reducer.compute_lr(test_loss.result().numpy(),test_accuracy.result().numpy(),optimizer.lr)
            #print("Reduce lr to {}".format(optimizer.lr))
        history.train_loss.append(train_loss.result())
        history.train_acc.append(train_accuracy.result())
        history.test_loss.append(test_loss.result())
        history.test_acc.append(test_accuracy.result())
        
        # TRAINING
        #train_ds=ori_train_ds.copy()
        if augmentation:
            aug_ds = train_ds.map(func_aug)
            train_iter = iter(aug_ds)
        else:
            train_iter=iter(train_ds)
        # iteration of batches

        trainable_list=[]
        for layer in range(len(model.variables)):
            if model.variables[layer].trainable:
                trainable_list.append(layer)

        for batch in range(num_train_batches):
            train_batch=next(train_iter)
            #train_batch=augmentation(train_batch[0],train_batch[1])
            # create gradientTape, use tape.gradient to compute gradient of certain variable.
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                predictions=model(train_batch[0],training=True)
                if num_classes==1:
                    loss=loss_object(train_batch[1],predictions)
                else:
                    loss = loss_object(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)
            gradients=tape.gradient(loss,model.trainable_variables)

            for layer in range(len(gradients)):
                gradients[layer]=gradients[layer]*masks[trainable_list[layer]]

            optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    weights=[]
    for layer in range(len(model.trainable_variables)):
        weights.append(tf.reshape(model.trainable_variables[layer],[-1]))
    weights=tf.concat(weights,0).numpy()
    nonzero=(np.product(weights.shape)-np.sum(abs(weights)==0))/np.product(weights.shape)

    history.history_weights.append(model.get_weights().copy())
    history.final_weights=model.get_weights().copy()
    history.nonzero=nonzero
    return history
import tensorflow_model_optimization as tfmot

def prune_model(model,mask,sparsity):
    ret_model=tf.keras.models.clone_model(model)
    weights=ret_model.get_weights()
    for layer in range(len(weights)):
        weights[layer] = weights[layer] * mask[layer]
    ret_model.set_weights(weights)
    ret_model=tfmot.sparsity.keras.prune_low_magnitude(model,
                    pruning_schedule=tfmot.sparsity.keras.pruning_sched.ConstantSparsity(sparsity,0,1,100))
    return ret_model

def regular_training(model, train_ds, test_ds, lr=1e-2, epoch=20, num_classes=1,
                  history=None, augmentation=True, opt='SGD', reduceLROnPlateau=False):
    if history == None:
        history = TrainHistory()
    history.epoch = epoch
    history.train_ds = train_ds
    history.test_ds = test_ds
    history.init_weights = model.get_weights().copy()

    # Training
    EPOCHS = epoch
    num_train_batches = tf.data.experimental.cardinality(train_ds)
    num_test_batches = tf.data.experimental.cardinality(test_ds)

    # loss and optimizer
    if num_classes == 1:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy()
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.0, nesterov=False)
    elif opt == 'ADAM':
        optimizer = tf.keras.optimizers.Adam(lr=lr)

    if reduceLROnPlateau:
        reducer = ReduceLROnPlateau(
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6
        )
    for epoch in range(EPOCHS+1):

        history.history_weights.append(model.get_weights().copy())
        gc.collect()
        optimizer.lr = lr
        if epoch > 80:
            optimizer.lr = lr * 1e-1
        if epoch > 120:
            optimizer.lr = lr * 1e-2
        if epoch > 160:
            optimizer.lr = lr * 1e-3
        if epoch > 180:
            optimizer.lr = lr * 0.5e-3

        # initialize dataset iterators
        train_iter = iter(train_ds)
        test_iter = iter(test_ds)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        if num_classes == 1:
            train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        else:
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        # test epoch (ON TEST DATA)
        for batch in range(num_test_batches):
            test_batch = next(test_iter)
            predictions = model(test_batch[0])
            if num_classes == 1:
                loss = loss_object(test_batch[1], predictions)
                test_accuracy(test_batch[1], predictions)
            else:
                loss = loss_object(tf.keras.utils.to_categorical(test_batch[1], num_classes=num_classes), predictions)
                test_accuracy(tf.keras.utils.to_categorical(test_batch[1], num_classes=num_classes), predictions)
            test_loss(loss)

        # test epoch (ON TRAIN DATA)
        for batch in range(num_train_batches):
            train_batch = next(train_iter)
            predictions = model(train_batch[0])
            if num_classes == 1:
                loss = loss_object(train_batch[1], predictions)
                train_accuracy(train_batch[1], predictions)
            else:
                loss = loss_object(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)
                train_accuracy(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes), predictions)
            train_loss(loss)

        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        # history.save('temp/dense_{}.history'.format(datetime.datetime.now().strftime('%m-%d-%Y-%H:%M:%S')))
        tf.print(template.format(epoch,
                                 np.round(train_loss.result(), 5),
                                 np.round(train_accuracy.result(), 5),
                                 np.round(test_loss.result(), 5),
                                 np.round(test_accuracy.result(), 5)))

        if reduceLROnPlateau:
            optimizer.lr = reducer.compute_lr(test_loss.result().numpy(), test_accuracy.result().numpy(), optimizer.lr)

        history.train_loss.append(train_loss.result())
        history.train_acc.append(train_accuracy.result())
        history.test_loss.append(test_loss.result())
        history.test_acc.append(test_accuracy.result())

        # TRAINING
        if augmentation:
            aug_ds = train_ds.map(func_aug)
            train_iter = iter(aug_ds)
        else:
            train_iter = iter(train_ds)
        # iteration of batches

        trainable_list = []
        for layer in range(len(model.variables)):
            if model.variables[layer].trainable:
                trainable_list.append(layer)

        for batch in range(num_train_batches):
            train_batch = next(train_iter)
            # train_batch=augmentation(train_batch[0],train_batch[1])
            # create gradientTape, use tape.gradient to compute gradient of certain variable.
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                predictions = model(train_batch[0], training=True)
                if num_classes == 1:
                    loss = loss_object(train_batch[1], predictions)
                else:
                    loss = loss_object(tf.keras.utils.to_categorical(train_batch[1], num_classes=num_classes),
                                       predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    weights = []
    for layer in range(len(model.trainable_variables)):
        weights.append(tf.reshape(model.trainable_variables[layer], [-1]))
    weights = tf.concat(weights, 0).numpy()
    nonzero = (np.product(weights.shape) - np.sum(abs(weights) == 0)) / np.product(weights.shape)

    history.history_weights.append(model.get_weights().copy())
    history.final_weights = model.get_weights().copy()
    history.nonzero = nonzero
    
    return history

def compute_mask(model, nonzero, init_model=None, random=False,exclude=[-1,-2],global_pruning=False):
    weights = model.get_weights()
    weights = tf.identity_n(weights)



    if init_model:
        for i in range(len(weights)):
            weights[i] = weights[i] - init_model.get_weights()[i]
    else:
        weights = model.get_weights()

    thresholds = []
    weights = tf.identity_n(weights)

    # if random pruning, set weights to random (Same as prun random entries)
    if random:
        for layer in range(len(weights)):
            weights[layer] = tf.random.uniform(weights[layer].shape)
    masks = tf.identity_n(weights)

    # computer thresholds for all layers
    for layer in range(len(weights)):
        weight_layer = tf.abs(tf.reshape(weights[layer], [-1]))
        t_layer = np.flip(np.sort(weight_layer))[
            int(min(np.product(weight_layer.shape) - 1, np.product(weight_layer.shape) * nonzero))]
        #print(np.flip(np.sort(weight_layer)),nonzero,t_layer)
        thresholds.append(t_layer)

    # if global, set threshold to global threshold
    if global_pruning:
        temp_weight = compute_weights(model, None)
        temp_threshold=np.flip(np.sort(tf.abs(temp_weight)))[
            int(min(np.product(temp_weight.shape)-1,np.product(temp_weight.shape)*nonzero))
        ]
        for layer in range(len(weights)):
            thresholds[layer]=temp_threshold

    exclude=[(i+len(weights)) % len(weights) for i in exclude]
    for layer in range(len(weights)):
        if layer in exclude:
            masks[layer] = tf.cast((abs(masks[layer]) >= 0), dtype=float)
        elif not model.variables[layer].trainable:
            masks[layer] = tf.cast((abs(masks[layer]) >= 0), dtype=float)
        elif nonzero==0:
            masks[layer] = tf.cast((abs(masks[layer]) > (thresholds[layer])), dtype=float)
        else:
            masks[layer] = tf.cast((abs(masks[layer]) >= (thresholds[layer])), dtype=float)
        #print(np.flip(np.sort(tf.abs(tf.reshape(masks[layer], [-1])))), nonzero, thresholds[layer])
    # if global_pruning:
    #     temp_weight = compute_weights(masks, None)
    #     print(np.sum(temp_weight==0),np.product(temp_weight.shape))

    # for layer in exclude:
    #     masks[layer] = tf.cast((abs(masks[layer]) >= 0), dtype=float)
    # masks[len(weights) - 2] = tf.cast((abs(masks[len(weights) - 2]) >= 0), dtype=float)
    # masks[len(weights) - 1] = tf.cast((abs(masks[len(weights) - 1]) >= 0), dtype=float)


    return masks


def mask_evaluate(model, masks, dataset,num_classes=10,set_init=0):
    num_batches = tf.data.experimental.cardinality(dataset)
    init_weights=model.get_weights().copy()

    # mask weights
    if set_init==0:
        weights=model.get_weights()
        if not masks == None:
            for layer in range(len(weights)):
                if model.variables[layer].trainable:
                    weights[layer] = weights[layer] * masks[layer]
        model.set_weights(weights)

    # loss and optimizer
    if num_classes==1:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy()

    data_iter = iter(dataset)

    loss = tf.keras.metrics.Mean(name='train_loss')
    if num_classes == 1:
        accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    else:
        accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    # test epoch (ON TEST DATA)
    for batch in range(num_batches):
        data_batch = next(data_iter)

        predictions = model(data_batch[0])

        if num_classes == 1:
            sample_loss = loss_object(data_batch[1], predictions)
            accuracy(data_batch[1], predictions)
        else:
            sample_loss = loss_object(tf.keras.utils.to_categorical(data_batch[1], num_classes=num_classes), predictions)
            accuracy(tf.keras.utils.to_categorical(data_batch[1], num_classes=num_classes), predictions)
        loss(sample_loss)
    model.set_weights(init_weights)
    return [loss.result(),accuracy.result()]


class LinearHistory:
    length = None
    train_acc = []
    proj_train_para = []
    proj_train_sign = []
    proj_test_para = []
    proj_test_sign = []
    parameter = []
    nonzero = []
    history_weights = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    def __init__(self):
        self.length = None
        self.test_acc = []
        self.train_acc = []
        self.proj_train_para = []
        self.proj_train_sign = []
        self.proj_test_para = []
        self.proj_test_sign = []
        self.parameter = []
        self.nonzero = []
        self.history_weights = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    @classmethod
    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

class ReduceLROnPlateau():
    def __init__(self,
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0.0,
               **kwargs):

        if factor >= 1.0:
          raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 20.0
        self.mode = mode
        self.monitor_op = None

    def compute_lr(self, loss,acc,lr):
        new_lr=lr.numpy()
        #print('CURRENT best  {} loss {}, lr {}'.format(self.best, loss, lr))
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
        if self.best>loss:
            #print('update best, old {} new {}, lr {} wait {}'.format(self.best, loss, lr,self.wait))
            self.best=loss
            self.wait=0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
              if lr > self.min_lr:
                new_lr = lr * self.factor
                new_lr = max(new_lr, self.min_lr)
                self.cooldown_counter = self.cooldown
                self.wait = 0
        print('LR= ',new_lr)
        return new_lr
    def in_cooldown(self):
        return self.cooldown_counter > 0
