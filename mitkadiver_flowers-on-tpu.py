#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q tensorflow_addons')
get_ipython().system('pip install -q efficientnet')


# In[ ]:


import math, sys, re, os, gc
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import efficientnet.tfkeras as efn

from tensorflow import keras

from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# In[ ]:


COMP_ENV_UNKNOWN = 0
COMP_ENV_CPU = 1
COMP_ENV_GPU = 2
COMP_ENV_TPU = 3

def get_computation_environment(verbose=1):
    """ Detect computational hardware.
        To use the selected distribution strategy:
        with strategy.scope:
            --- define your (Keras) model here ---
        
        For distributed computing, the batch size and learning rate need to be adjusted:
        global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync # num replcas is 8 on a single TPU or N when runing on N GPUs.
        learning_rate = LEARNING_RATE * strategy.num_replicas_in_sync
    """
    calculator = COMP_ENV_UNKNOWN
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")
    
    # Select appropriate distribution strategy for hardware
    if tpu:
        calculator = COMP_ENV_TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    elif len(gpus) > 0:
        calculator = COMP_ENV_GPU
        strategy = tf.distribute.MirroredStrategy(gpus) # this works for 1 to multiple GPUs
    else:
        calculator = COMP_ENV_CPU
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        
    
    if verbose == 1 or verbose == 2:
        if calculator == COMP_ENV_TPU:
            print('Running on TPU ', tpu.master())
        elif calculator == COMP_ENV_GPU:
            print('Running on ', len(gpus), ' GPU(s) ')
        elif calculator == COMP_ENV_CPU:
            print('Running on CPU')
        else:
            print("Running on unknown environment")
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    return calculator, strategy


# In[ ]:


calculator, strategy = get_computation_environment(verbose=2)


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# In[ ]:


SKIP_VALIDATION = False
use_half_splitted_dataset = False
GLOBAL_VERBOSE = 2

img_size = 512
lr_epochs_profile = {'RAMPUP':3, 'SUSTAIN':2, 'DECAY':35}
# lr_epochs_profile = {'RAMPUP':1, 'SUSTAIN':1, 'DECAY':1}

IMAGE_SIZE = [img_size, img_size] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = lr_epochs_profile['RAMPUP'] + lr_epochs_profile['SUSTAIN'] + lr_epochs_profile['DECAY']

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

NUM_OF_CLASSES = len(CLASSES)

#------------------------------
# Custom LR schedule
#------------------------------
if calculator == COMP_ENV_TPU:
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
else:
    LR_MAX = 0.00005

LR_MIN = LR_MAX / 10.0
LR_START = LR_MAX / 3.0
LR_START_AMPLITUDE = LR_MAX / 4.0
LR_RAMPUP_EPOCHS = lr_epochs_profile['RAMPUP']
LR_SUSTAIN_EPOCHS = lr_epochs_profile['SUSTAIN']
LR_EXP_DECAY = .8

#------------------------------

@tf.function
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

early_stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='auto',
    restore_best_weights=True)


# In[ ]:


# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[np.argmax(label)]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# In[ ]:


def show_training_curves(training_history):
    display_training_curves(training_history.history['loss'], training_history.history['val_loss'], 'loss', 211)
    display_training_curves(training_history.history['sparse_categorical_accuracy'], training_history.history['sparse_categorical_accuracy'], 'accuracy', 212)

def get_cmat_score_precision_recall__one_hot(models, num_of_classes, cmdataset, num_of_images):
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    cm_correct_labels = np.array(list(map(np.argmax, next(iter(labels_ds.batch(num_of_images))).numpy()))) # get everything as one batch
    
    cm_probabilities = np.zeros(num_of_classes, dtype=float)
    for model in models:
        probabilities = model.predict(images_ds)
        cm_probabilities = cm_probabilities + probabilities
        
    cm_predictions = np.argmax(cm_probabilities, axis=-1)

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(num_of_classes))
    score = f1_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
    return cmat, score, precision, recall

def get_cmat_score_precision_recall(models, alphas, num_of_classes, cmdataset, num_of_images, verbose=0):
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    cm_correct_labels = next(iter(labels_ds.batch(num_of_images))).numpy() # get everything as one batch
    
    cm_probabilities = np.zeros(num_of_classes, dtype=float)
    for i in range(len(models)):
        model = models[i]
        alpha = alphas[i]
        probabilities = alpha * model.predict(images_ds)
        cm_probabilities = cm_probabilities + probabilities
        
    cm_predictions = np.argmax(cm_probabilities, axis=-1)
    
    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(num_of_classes))
    score = f1_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(num_of_classes), average='macro')
    cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
    return cmat, score, precision, recall

def show_confusion_matrix(models, num_of_classes, cmdataset, num_of_images, verbose=0):
    cmat, score, precision, recall = get_cmat_score_precision_recall(models)
    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
    display_confusion_matrix(cmat, score, precision, recall)


# In[ ]:


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example, one_hot=False):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    if one_hot:
        label = tf.one_hot(label, depth=len(CLASSES))
    return image, label # returns a dataset of (image, label) pairs
    
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)
    
def force_image_sizes(dataset, image_size):
    # explicit size needed for TPU
    reshape_images = lambda image, label: (tf.reshape(image, [*image_size, 3]), label)
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)
    return dataset

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, one_hot_class):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.8, 1)
#     image = tf.image.random_jpeg_quality(image, 80, 100)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1)
    return image, one_hot_class

def tf_tut_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    image = tf.image.random_saturation(image, 0.8, 1)
    
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, 0.8, 1)
    
    resize_factor = (img_size // 5) * 6
    image = tf.image.resize_with_crop_or_pad(image, resize_factor, resize_factor)
    image = tf.image.random_crop(image, size=[*IMAGE_SIZE, 3])
    
    tf.image.random_jpeg_quality(image, 20, 100)
    return image,label
    
def get_training_dataset(dataset):
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def calc_datasets_parameters(training_filenames, validation_filenames, test_filenames, batch_size):
    NUM_TRAINING_IMAGES = count_data_items(training_filenames)
    NUM_VALIDATION_IMAGES = count_data_items(validation_filenames)
    NUM_TEST_IMAGES = count_data_items(test_filenames)
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // batch_size
    print('BASE Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
    return NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES, STEPS_PER_EPOCH

def get_raw_half_splitted_training_datasets(training_filenames):
    half_of_images = count_data_items(training_filenames) // 2
    dataset = load_dataset(training_filenames, labeled=True)
    dataset.shuffle(2048)
    ds_1 = dataset.take(half_of_images)
    ds_2 = dataset.skip(half_of_images)
    print("Splitted on 2 datasets with {} images".format(half_of_images))
    return ds_1, ds_2

def prepare_weights(trainingdataset, num_of_classes, calculator, one_hot=False, verbose=0):
    counters = np.zeros(num_of_classes, dtype=float)
    total_samples = 0.0
    
    for batch_data in trainingdataset:
        _, labels = batch_data
        if one_hot:
            np.add(counters, labels.numpy(), out=counters)
        else:
            counters[labels.numpy()] += 1
        total_samples += 1

    weights = total_samples / (num_of_classes * counters)
    weights = weights / weights.min()
    
    if calculator == COMP_ENV_TPU:
        classweights = weights.tolist()
    else:
        classweights = dict(enumerate(weights))
    
    if verbose == 1 or verbose ==2:
        print("Weihgting. Counters, min : {},  max : {}".format(counters.min(),counters.max()))
        
    if verbose ==2:
        print('Ordered counts:')
        print(np.sort(counters))
        print('Weights:')
        print(classweights)
    return classweights


# In[ ]:


def get_model_EfficientNetB7():
       return efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

def get_model_DenseNet201():
       return keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

def get_model_Xception():
       return keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

def get_model_ResNet50V2():
       return keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])


# In[ ]:


def create_model(pretrained_model, num_of_classes):
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(NUM_OF_CLASSES, activation=None, name='Output_features'),   # , kernel_regularizer=keras.regularizers.l2(0.05)
        keras.layers.Softmax()
    ])
    return model

def create_Ranger_optimizer(lr, min_lr, total_steps):
    radam = tfa.optimizers.RectifiedAdam(
        lr=lr,
        total_steps=total_steps,
        warmup_proportion=0.1,
        min_lr=min_lr,
    )
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    return ranger


def create_model_1(num_of_classes):
    model = create_model(get_model_EfficientNetB7(), num_of_classes)
    
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['sparse_categorical_accuracy']
    )
    return model

def create_model_2(num_of_classes):
    model = create_model(get_model_DenseNet201(), num_of_classes)
    
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['sparse_categorical_accuracy']
    )
    return model


# In[ ]:


G_NUM_TRAINING_IMAGES, G_NUM_VALIDATION_IMAGES, G_NUM_TEST_IMAGES, G_STEPS_PER_EPOCH = calc_datasets_parameters(
    TRAINING_FILENAMES, VALIDATION_FILENAMES, TEST_FILENAMES, BATCH_SIZE)


# In[ ]:


if use_half_splitted_dataset:
    raw_training_ds_1, raw_training_ds_2 = get_raw_half_splitted_training_datasets(TRAINING_FILENAMES)
    steps_per_epoch = G_STEPS_PER_EPOCH // 2

    classweights_1 = prepare_weights(raw_training_ds_1, NUM_OF_CLASSES, calculator, one_hot=False, verbose=0)
    classweights_2 = prepare_weights(raw_training_ds_2, NUM_OF_CLASSES, calculator, one_hot=False, verbose=0)

    training_dataset_1 = get_training_dataset(raw_training_ds_1)
    training_dataset_2 = get_training_dataset(raw_training_ds_2)
else:
    raw_training_dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    steps_per_epoch = G_STEPS_PER_EPOCH

    classweights_1 = prepare_weights(raw_training_dataset, NUM_OF_CLASSES, calculator, one_hot=False, verbose=0)
    classweights_2 = classweights_1

    training_dataset_1 = get_training_dataset(raw_training_dataset)
    training_dataset_2 = training_dataset_1


# In[ ]:


validation_dataset = get_validation_dataset(ordered=True)
cm_dataset = get_validation_dataset(ordered=True)


# In[ ]:


print("Image size : {}".format(IMAGE_SIZE))
print("EPOCHS : {}".format(EPOCHS))
print("Steps per epoch : {}".format(steps_per_epoch))


# In[ ]:


# numpy and matplcm_datasetotlib defaults
np.set_printoptions(threshold=sys.maxsize, linewidth=80)


# In[ ]:


gc.collect()
with strategy.scope():
    model_1 = create_model_1(NUM_OF_CLASSES)
model_1.summary()


# In[ ]:


history_1 = model_1.fit(
    training_dataset_1,
    validation_data=validation_dataset,
    steps_per_epoch=int(steps_per_epoch),
    epochs=EPOCHS,
    class_weight = classweights_1,
    callbacks=[lr_callback, early_stop_callback],
    verbose=GLOBAL_VERBOSE,
)

# model_1.save('model_1.h5')


# In[ ]:


# show_training_curves(history_1)


# In[ ]:


gc.collect()
with strategy.scope():
    model_2 = create_model_2(NUM_OF_CLASSES)
model_2.summary()


# In[ ]:


history_2 = model_2.fit(
    training_dataset_2,
    validation_data=validation_dataset,
    steps_per_epoch=int(steps_per_epoch),
    epochs=EPOCHS,
    class_weight = classweights_1,
    callbacks=[lr_callback, early_stop_callback],
    verbose=GLOBAL_VERBOSE,
)

# model_2.save('model_2__DenseNet201.h5')


# In[ ]:


# keras.utils.plot_model(model_1, 'model.png', show_shapes=True)


# In[ ]:


# keras.utils.plot_model(model_2, 'model.png', show_shapes=True)


# In[ ]:


if not SKIP_VALIDATION:
    cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    cm_correct_labels = next(iter(labels_ds.batch(G_NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
    m1 = model_1.predict(images_ds)
    m2 = model_2.predict(images_ds)
    scores = []
    N = 100
    for alpha in np.linspace(0, 1, N):
        cm_probabilities = alpha * m1 + (1 - alpha) * m2
        cm_predictions = np.argmax(cm_probabilities, axis=-1)
        scores.append(f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro'))
        
    plt.plot(scores)
    best_alpha = np.argmax(scores) / N
    cm_probabilities = best_alpha * m1 + (1 - best_alpha) * m2
    cm_predictions = np.argmax(cm_probabilities, axis=-1)
else:
    best_alpha = 0.55

print(best_alpha)

alpha_1 = best_alpha
alpha_2 = 1.0 - alpha_1


# In[ ]:


alphas = [alpha_1, alpha_2]
cmat, score_1, precision_1, recall_1 = get_cmat_score_precision_recall([model_1], [1.0], NUM_OF_CLASSES, cm_dataset, G_NUM_VALIDATION_IMAGES)
cmat, score_2, precision_2, recall_2 = get_cmat_score_precision_recall([model_2], [1.0], NUM_OF_CLASSES, cm_dataset, G_NUM_VALIDATION_IMAGES)
cmat, score_12, precision_12, recall_12 = get_cmat_score_precision_recall([model_1, model_2], alphas, NUM_OF_CLASSES, cm_dataset, G_NUM_VALIDATION_IMAGES)

print("MODEL___1  F1 : {}, precision : {}, Recall : {}".format(score_1, precision_1, recall_1))
print("MODEL___2  F1 : {}, precision : {}, Recall : {}".format(score_2, precision_2, recall_2))
print("MODEL_1+2  F1 : {}, precision : {}, Recall : {}".format(score_12, precision_12, recall_12))


# In[ ]:


print('Computing predictions...')
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.
test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities_1 = alpha_1 * model_1.predict(test_images_ds)
probabilities_2 = alpha_2 * model_2.predict(test_images_ds)

probabilities = probabilities_1 + probabilities_2

predictions = np.argmax(probabilities, axis=-1)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(G_NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
get_ipython().system('head submission.csv')


# In[ ]:




