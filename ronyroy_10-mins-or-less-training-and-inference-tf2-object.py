#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U --pre tensorflow=="2.2.0" # totally unncessary.. but hey no harm in double checking...')


# In[2]:


import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('models').exists():
    get_ipython().system('git clone --depth 1 https://github.com/tensorflow/models')


# In[3]:


cd models/research


# In[4]:


# give it all absolute paths if this doesnt work...
# for any reason
get_ipython().system('protoc object_detection/protos/*.proto --python_out=.')
get_ipython().system('cp object_detection/packages/tf2/setup.py .')
get_ipython().system('python -m pip install .')


# In[5]:


import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
import pandas as pd

from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import colab_utils
# this was for the annotator can be done away with in this kaggle env
from object_detection.builders import model_builder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def load_image_into_numpy_array(path):
    ''
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.2)
    
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


# In[7]:


train_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')


# In[8]:


train_df['bboxs'] = train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=','))


# In[9]:


train_df.head()


# In[10]:


train_df.width.nunique(),train_df.height.nunique()


# In[11]:


def xyxy_to_yxyx(a): 
    #     a_row is  ndarray..
    a[2] = a[0]+a[2]
    a[3] = a[1]+a[3]
    a[0],a[1],a[2],a[3] = a[1],a[0],a[3],a[2] # xyxy to yxyx here..
    a = a/1024
           
    return a.reshape(1,4)


# In[12]:


train_df['bbox_adj'] = '' 


# In[13]:


train_df['bbox_adj'] = train_df['bboxs'].map(xyxy_to_yxyx)


# In[14]:


image_name_and_bboxes = {} # placeholder dict for the images and coressponding bboxes


# In[15]:


get_ipython().run_cell_magic('time', '', "for i in range(len(train_df)):\n    if train_df.iloc[i]['image_id'] in image_name_and_bboxes.keys():\n        image_name_and_bboxes[train_df.iloc[i]['image_id']] = np.vstack((image_name_and_bboxes[train_df.iloc[i]['image_id']],train_df.iloc[i]['bbox_adj']))\n    else:\n        image_name_and_bboxes[train_df.iloc[i]['image_id']] = train_df.iloc[i]['bbox_adj']")


# In[16]:


# Hack
# the setup is geared for few shot object recognition so pick images with few wheat heads


# In[17]:


val_counts_image_id = pd.DataFrame(train_df['image_id'].value_counts())


# In[18]:


val_counts_image_id.head()


# In[19]:


val_counts_image_id['bbox_count'] = val_counts_image_id['image_id'] 


# In[20]:


val_counts_image_id['image_id']  = val_counts_image_id.index


# In[21]:


val_counts_image_id.reset_index(drop = True, inplace = True)


# In[22]:


val_counts_image_id.head() # 


# In[23]:


cond_1 = val_counts_image_id['bbox_count']==1
cond_2 = val_counts_image_id['bbox_count']==2
cond_3 = val_counts_image_id['bbox_count']==3


# In[24]:


val_counts_image_id['image_id'][cond_1 | cond_2]


# In[25]:


image_names_list = val_counts_image_id['image_id'][cond_1 | cond_2].tolist()


# In[26]:


image_names_list # only images with one or 2  or 3 wheat heads higher numbers dont converge well... for now...


# In[27]:


# Place holders
train_image_dir = '/kaggle/input/global-wheat-detection/train/'
image_names_list = image_names_list = val_counts_image_id['image_id'][cond_1 | cond_2].tolist() 

NUM_IMAGES = len(image_names_list)

train_images_np = []
list_of_bboxes = []


# In[28]:


train_df.shape


# In[29]:


get_ipython().run_cell_magic('time', '', "for i in range(NUM_IMAGES):\n    image_path = os.path.join(train_image_dir, image_names_list[i] + '.jpg')\n    train_images_np.append(load_image_into_numpy_array(image_path))")


# In[30]:


get_ipython().run_cell_magic('time', '', 'for i in range(NUM_IMAGES):\n    list_of_bboxes.append(image_name_and_bboxes[image_names_list[i]])')


# In[31]:


# By convention, our non-background classes start counting at 1.  Given
# that we will be predicting just one class, we will therefore assign it a
# `class id` of 1.
gt_boxes = list_of_bboxes # fewer breakdowsn this way
wheat_class_id = 1
num_classes = 1

category_index = {wheat_class_id: {'id': wheat_class_id, 'name': 'wheat'}}


# In[32]:


# Convert class labels to one-hot; convert everything to tensors.
# The `label_id_offset` here shifts all classes by a certain number of indices;
# we do this here so that the model receives one-hot labels where non-background
# classes start counting at the zeroth index.  This is ordinarily just handled
# automatically in our training binaries, but we need to reproduce it here.
label_id_offset = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []

for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
    train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
      train_image_np, dtype=tf.float32), axis=0))
    
    gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
    
    zero_indexed_groundtruth_classes = tf.convert_to_tensor(
      np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
    
    gt_classes_one_hot_tensors.append(tf.one_hot(
      zero_indexed_groundtruth_classes, num_classes))
    
print('Done prepping data.')


# In[33]:


# Download the checkpoint and put it into models/research/object_detection/test_data/

get_ipython().system('wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz')
get_ipython().system('tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz')
get_ipython().system('mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint /kaggle/working/models/research/object_detection/test_data/')


# In[34]:


tf.keras.backend.clear_session()

print('Building model and restoring weights for fine-tuning...', flush=True)
num_classes = 1
pipeline_config = '/kaggle/working/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = '/kaggle/working/models/research/object_detection/test_data/checkpoint/ckpt-0'


# In[35]:



# Load pipeline config and build a detection model.
#
# Since we are working off of a COCO architecture which predicts 90
# class slots by default, we override the `num_classes` field here to be just
# one (for our new rubber ducky class).
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(
      model_config=model_config, is_training=True)

# Set up object-based checkpoint restore --- RetinaNet has two prediction
# `heads` --- one for classification, the other for box regression.  We will
# restore the box regression head but initialize the classification head
# from scratch (we show the omission below by commenting out the line that
# we would add if we wanted to restore both heads)
fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(checkpoint_path).expect_partial()

# Run model through a dummy image so that variables are created
image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')


# In[36]:


get_ipython().run_cell_magic('time', '', '\n# Select variables in top layers to fine-tune.\ntrainable_variables = detection_model.trainable_variables\nto_fine_tune = []\nprefixes_to_train = [\n  \'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead\',\n  \'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead\']\nfor var in trainable_variables:\n    if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):\n        to_fine_tune.append(var)\n\n# Set up forward + backward pass for a single train step.\ndef get_model_train_step_function(model, optimizer, vars_to_fine_tune):\n    """Get a tf.function for training step."""\n\n    # Use tf.function for a bit of speed.\n    # Comment out the tf.function decorator if you want the inside of the\n    # function to run eagerly.\n    @tf.function\n    def train_step_fn(image_tensors,\n                    groundtruth_boxes_list,\n                    groundtruth_classes_list):\n        """A single training iteration.\n\n        Args:\n          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.\n            Note that the height and width can vary across images, as they are\n            reshaped within this function to be 640x640.\n          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type\n            tf.float32 representing groundtruth boxes for each image in the batch.\n          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]\n            with type tf.float32 representing groundtruth boxes for each image in\n            the batch.\n\n        Returns:\n          A scalar tensor representing the total loss for the input batch.\n        """\n        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)\n        model.provide_groundtruth(\n            groundtruth_boxes_list=groundtruth_boxes_list,\n            groundtruth_classes_list=groundtruth_classes_list)\n        with tf.GradientTape() as tape:\n            preprocessed_images = tf.concat(\n              [detection_model.preprocess(image_tensor)[0]\n               for image_tensor in image_tensors], axis=0)\n            prediction_dict = model.predict(preprocessed_images, shapes)\n            losses_dict = model.loss(prediction_dict, shapes)\n            total_loss = losses_dict[\'Loss/localization_loss\'] + losses_dict[\'Loss/classification_loss\']\n            gradients = tape.gradient(total_loss, vars_to_fine_tune)\n            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))\n        return total_loss\n\n    return train_step_fn\n\n')


# In[37]:


get_ipython().run_cell_magic('time', '', "tf.keras.backend.set_learning_phase(True)\n\n# These parameters can be tuned; since our training set has 5 images\n# it doesn't make sense to have a much larger batch size, though we could\n# fit more examples in memory if we wanted to.\nbatch_size = 4\nlearning_rate = 0.001\nnum_batches = 200\n\n# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)\noptimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,  decay=learning_rate / num_batches)\ntrain_step_fn = get_model_train_step_function(\n    detection_model, optimizer, to_fine_tune)\n\n\nprint('Start fine-tuning!', flush=True)\nfor idx in range(num_batches):\n  # Grab keys for a random subset of examples\n  all_keys = list(range(len(train_images_np)))\n  random.shuffle(all_keys)\n  example_keys = all_keys[:batch_size]\n\n  # Note that we do not do data augmentation in this demo.  If you want a\n  # a fun exercise, we recommend experimenting with random horizontal flipping\n  # and random cropping :)\n  gt_boxes_list = [gt_box_tensors[key] for key in example_keys]\n  gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]\n  image_tensors = [train_image_tensors[key] for key in example_keys]\n\n  # Training step (forward pass + backwards pass)\n  total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)\n\n  if idx % 10 == 0:\n    print('batch ' + str(idx) + ' of ' + str(num_batches)\n    + ', loss=' +  str(total_loss.numpy()), flush=True)\n\nprint('Done fine-tuning!')")


# In[38]:


test_image_dir = '/kaggle/input/global-wheat-detection/test/'

test_images_np = []


# In[39]:


os.listdir(test_image_dir)


# In[40]:


for i in range(1): # test against one
  image_path = os.path.join(test_image_dir, '348a992bb.jpg')
  test_images_np.append(np.expand_dims(
      load_image_into_numpy_array(image_path), axis=0))

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor):
  """Run detection on an input image.

  Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  return detection_model.postprocess(prediction_dict, shapes)


# In[41]:


# Note that the first frame will trigger tracing of the tf.function, which will
# take some time, after which inference should be fast.

label_id_offset = 1
for i in range(len(test_images_np)):
    input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
    detections = detect(input_tensor)

    plot_detections(
      test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32)
      + label_id_offset,
      detections['detection_scores'][0].numpy(),
      category_index, figsize=(15, 20), image_name="inf_348a992bb.jpg")


# In[42]:


#Import library
from IPython.display import Image# Load image from local storage
Image(filename = 'inf_348a992bb.jpg', width = 512, height = 512)


# In[ ]:




