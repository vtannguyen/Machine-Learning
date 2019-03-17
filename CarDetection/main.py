import argparse
import math
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import h5py
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Concatenate
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, \
                                    yolo_boxes_to_corners, \
                                    preprocess_true_boxes, \
                                    yolo_loss, yolo_body, \
                                    space_to_depth_x2, \
                                    space_to_depth_x2_output_shape

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """

    # Compute box scores
    box_scores = box_confidence * box_class_probs

    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # Create a filtering mask based on "box_class_scores" by using "threshold".
    filtering_mask = box_class_scores > threshold

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def create_yolo_model(input_shape):
    """ create yolo model """
    X_input = Input(shape=input_shape)

    # 1st stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # 2nd stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # 3rd stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 4th stage
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 5th stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # 6th stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 7th stage
    X = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 8th stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    # 9th stage
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 10th stage
    X = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 11th stage
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 12th stage
    X = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 13th stage
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X_shortcut = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_shortcut)

    # 14th stage
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 15th stage
    X = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 16th stage
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 17th stage
    X = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 18th stage
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # 19th stage
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    # shortcut stage
    X_shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1)(X_shortcut)
    X_shortcut = LeakyReLU(alpha=0.3)(X_shortcut)
    X_shortcut = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(X_shortcut)

    # 20th stage
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    X = Concatenate(axis=-1)([X, X_shortcut])
    X = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=-1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Conv2D(filters=425, kernel_size=(1, 1), strides=(1, 1), padding='same')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def image_name(numb):
    """ Convert number to string of form 000x """
    assert (numb<10000), "number is too large"
    if numb < 10:
        return "000" + str(numb)
    elif numb >= 10 & numb < 100:
        return "00" + str(numb)
    elif numb >= 100 & numb < 1000:
        return "0" + str(numb)
    else:
        return str(numb)


def yolo_corners_to_boxes(boxes, image_shape):
    """ convert yolo corner to yolo center box

    Parameters:
    -----------
    boxes: array
        contain of boxes with shape of [x_top_left, y_top_left, x_bottom_right, y_bottom_y]
    image_shape: tuple
        shape of the image

    Returns:
    --------
    converted_boxes: array
        contain of boxes with shape of [x_center, y_center, width, height]
    """
    converted_boxes = np.zeros(boxes.shape)

    converted_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0 / image_shape[0]
    converted_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0 / image_shape[1]
    converted_boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) / image_shape[0]
    converted_boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) / image_shape[1]

    return converted_boxes


def process_output_boxes(boxes, classes, image_shape):
    """ Create true_boxes of shape [m, x, y, w, h, class] for training yolo model

    Parameters:
    -----------
    boxes: array
        array of boxes with shape of [x_top_left, y_top_left, x_bottom_right, y_bottom_y]
    classes: array
        corresponding class of boxes
    image_shape: tuple
        shape of original image

    Returns:
    --------
    true_boxes of shape [m, x, y, w, h, class]
    """
    converted_boxes = yolo_corners_to_boxes(boxes, image_shape)
    print(converted_boxes.shape)
    print(classes.shape)
    true_boxes = np.append(converted_boxes, np.reshape(classes, (classes.shape[0], 1)), axis=1)

    return np.expand_dims(true_boxes, axis=0)


def generate_true_boxes(numb_image, yolo_model_path, true_boxes_path):
    """ Generate ground truth boxes automatically using trained yolo model

    Parameters:
    -----------
    numb_image: integer
        number of image to process
    yolo_model_path: string
        location of trained yolo model

    Returns:
    --------
    """
    yolo_model = load_model(yolo_model_path)

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    sess = K.get_session()

    list_boxes = []
    for i in range(1, numb_image + 1):
        image_tmp, image_data_tmp = preprocess_image("images/" + image_name(i) + ".jpg", model_image_size=(608, 608))
        out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes],
                                                      feed_dict={yolo_model.input: image_data_tmp, K.learning_phase(): 0})

        boxes_tmp = process_output_boxes(out_boxes, out_classes, image_shape)

        boxes_tmp_expand = np.zeros([1, 10, 5])
        boxes_tmp_expand[:, 0:boxes_tmp.shape[1], :] = boxes_tmp

        if len(list_boxes) == 0:
            list_boxes = boxes_tmp_expand
        else:
            list_boxes = np.append(list_boxes, boxes_tmp_expand, axis=0)

    h5f = h5py.File(true_boxes_path, 'w')
    h5f.create_dataset('data', data=list_boxes)
    h5f.close()


def choose_trainning_set(ratio, dataset):
    """ Separate training set and test set from dataset

    Parameters:
    -----------
    ratio : float
        percentage of train examples over total number of examples
    dataset : dictionary
        X : array - input data of shape [m, ...]
        Y : array - output data of shape [m, ...]

    Returns:
    --------
    X_train : array
        training input data
    Y_train : array
        training output data
    X_test : array
        testing input data
    Y_test : array
        testing output data
    """
    X, Y = dataset
    m = X.shape[0] # number of examples
    m_train = int(m*ratio)
    X_train = X[0:m_train, ...]
    Y_train = y[0:m_train, ...]
    X_test = X[m_train:m, ...]
    Y_test = y[m_train:m, ...]

    return X_train, Y_train, X_test, Y_test


def load_data(numb_image, true_boxes_path):
    """ Load images and prepocess images, load ground truth boxes

    Parametes:
    ----------
    numb_image : integer
        number of images used for training
    true_boxes_path : string
        location of true boxes

    Returns:
    --------
    dataset : dictionary
        X: processed images
        Y: corresponding ground truth boxes
    """
    dataset = {}

    # Preprocess image data
    image_data = [] # To store all processed image data
    for i in range(1, numb_image + 1):
        image_tmp, image_data_tmp = preprocess_image("images/" + image_name(i) + ".jpg", model_image_size=(608, 608))
        if len(image_data) == 0:
            image_data = image_data_tmp
        else:
            image_data = np.append(image_data, image_data_tmp, axis=0)
    dataset['X'] = image_data

    # Load true_boxes data
    h5f = h5py.File(true_boxes_path, 'r')
    true_boxes_data = h5f['data'][:]
    data = np.zeros(true_boxes_data[0:numb_image, :, :].shape, dtype='float32')
    data[:] = true_boxes_data[0:numb_image, :, :]
    dataset['Y'] = data

    return dataset


def random_mini_batches(X, Y, mini_batch_size=64):
    """ Creates a list of random minibatches from (X, Y)

    Parameters:
    -----------
    X : array
        input data, of shape (m, Hi, Wi, Ci)
    Y : array
        list of ground truth box, of shape (m, numb_boxes, numb_box_params)
    mini_batch_size : integer
        size of the mini-batches

    Returns:
    --------
    mini_batches : array
        list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def train_yolo_model(numb_image=10,
                     learning_rate=0.00005,
                     num_epochs=100,
                     minibatch_size=5,
                     print_cost=True,
                     model_path="./model_data/yolo_trained_model.h5"):
    """ Create and train yolo model
    Parameters:
    -----------
    numb_image : int
        number of images used for training
    learning_rate : float
        learning rate of the optimization
    num_epochs : int
        number of epochs of the optimization loop
    minibatch_size : int
        size of a minibatch
    print_cost : boolean
        True to print the cost every 5 epochs
    model_path : string
        location to save trained model

    Returns:
    --------
    costs : list
        list of cost values
    """
    costs = []  # To keep track of the cost

    # Load data
    image_size = (608, 608)
    dataset = load_data(numb_image, "model_data/true_boxes.h5")

    # Create model
    X = Input(batch_shape=[None, 608, 608, 3], dtype=tf.float32)
    Y1 = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32) # true boxes
    Y2 = tf.placeholder(shape=[None, 19, 19, 5, 1], dtype=tf.float32) # detector mask
    Y3 = tf.placeholder(shape=[None, 19, 19, 5, 5], dtype=tf.float32) # matching true boxes
    yolo_m = yolo_body(inputs=X, num_anchors=len(anchors), num_classes=len(class_names))

    args = [yolo_m.output, Y1, Y2, Y3]
    cost = yolo_loss(args, anchors, len(class_names))

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Run session to train network
    with tf.Session() as sess:
        # Run initialization
        sess.run(init)

        # # Run training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(numb_image / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(dataset['X'], dataset['Y'], minibatch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Preprocess true boxes
                detectors_mask = []
                matching_true_boxes = []
                for i in range(0, minibatch_Y.shape[0]):
                    detectors_mask_tmp, matching_true_boxes_tmp = preprocess_true_boxes(minibatch_Y[i, :, :], anchors, image_size)
                    if len(detectors_mask) == 0:
                        detectors_mask = np.expand_dims(detectors_mask_tmp, axis=0)
                    else:
                        detectors_mask = np.append(detectors_mask, np.expand_dims(detectors_mask_tmp, axis=0), axis=0)

                    if len(matching_true_boxes) == 0:
                        matching_true_boxes = np.expand_dims(matching_true_boxes_tmp, axis=0)
                    else:
                        matching_true_boxes = np.append(matching_true_boxes, np.expand_dims(matching_true_boxes_tmp, axis=0), axis=0)

                # Run the session to execute the optimizer and the cost
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X,
                                                                      Y1: minibatch_Y,
                                                                      Y2: detectors_mask,
                                                                      Y3: matching_true_boxes})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        yolo_m.save_weights(model_path)

        return costs


yolo_model = load_model("model_data/yolo_trained_model.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
sess = K.get_session()


def predict(model_path, image_file):
    """ Runs the graph stored in "model_path" to predict boxes for "image_file".
    Prints and plots the preditions.

    Parameters:
    ----------
    model_path : string
        location of the YOLO graph
    image_file : string
        name of an image stored in the "images" folder.

    Returns:
    --------
    out_scores : tensor of shape (None, )
        scores of the predicted boxes
    out_boxes : tensor of shape (None, 4)
        coordinates of the predicted boxes
    out_classes : tensor of shape (None, )
        class index of the predicted boxes
    """
    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes


# generate true boxes for training the yolo model
# generate_true_boxes(numb_image=30, yolo_model_path="model_data/yolo.h5", true_boxes_path="model_data/true_boxes.h5")

# train the yolo model
cost = train_yolo_model(numb_image=30, num_epochs=5, minibatch_size=10)
plt.plot(cost)
plt.show()

# predict result using trained model
# out_scores, out_boxes, out_classes = predict("model_data/yolo_trained_model.h5", "test.jpg")
