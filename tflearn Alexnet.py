import os.path
import numpy
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import errno


'''
Here I write how I had to assemble 2 models with the same shape in tflearn
I had to repeat the same code with different variable names so that tensorflow didn't have any errors
'''


'''
Alexnet:

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

Tflearn code referenced from
https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
'''

def getProjectFolder():
    ProjectFolder = os.path.join(os.path.expanduser('~'),'my_project')   
    return ProjectFolder

def MakeLogFile(filename):
    halfpath=os.path.join(getProjectFolder(), 'logs')
    fullpath = os.path.join(halfpath, filename)
    return fullpath

def MakeModelPath(filename):
    halfpath=os.path.join(getProjectFolder(), 'models')
    fullpath = os.path.join(halfpath, filename)
    return fullpath

def getDataPath(mode = 'grayscale_resize_1to10', which = 'first'):
    if which == 'first' or which == 'second':
        data_path = os.path.join(getProjectFolder(),'Videos', 'numpy', '{}_{}class_vector.npy'.format(mode, which))
    else:
        raise ValueError('Invalid which = {}'.format(which))
    return data_path

# get X, Y, test_x, test_y ready
def ReadyData(data, test_size = 1000, do_shuffle=True):
    if do_shuffle:
        numpy.random.shuffle(data)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
    X,Y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    X = numpy.array(list(X))
    Y = numpy.array(list(Y))
    test_x = numpy.array(list(test_x))
    test_y = numpy.array(list(test_y))
    return X,Y,test_x,test_y

# Load X, Y, test_x, test_y
def LoadData(mode = 'grayscale_resize_1to10', which='first', split_test=True, test_size=1000, do_shuffle=True):
    print("Loading numpy array from file")
    # ## Load after first time
    input_path = getDataPath(mode= mode, which=which)
    if os.path.exists(input_path):
        data = numpy.load(input_path)
        print("Done")
        if split_test:
            X,Y,test_x,test_y = ReadyData(data, test_size=test_size, do_shuffle=do_shuffle)
            return X,Y,test_x,test_y
        else:
            return data
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_path)

### To check the tensorboard log
def print_log_instructions():
    print("To be able to see Tensorboard in your local machine after training on a server")
    print("    1. exit current server session")
    print("    2. connect again with the following command:")
    print("        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]")
    print("    3. execute in terminal")
    print("        tensorboard --logdir='{}'".format(MakeLogFile('')))
    print("    4. on local machine, open browser on:")
    print("        http://127.0.0.1:16006")

#### Model defining
## I have to set the network with different variable names or tensorflow has errors
def Convolutional_Model_Alexnet_first(img_height,img_width,color_channels, n_classes, run_id='', first_graph=tf.Graph()):
    # data size
    pixels = img_height*img_width
    batch_size = 64
    # pool window sizes
    pool_1_window_size = 3
    pool_2_window_size = 3
    pool_3_window_size = 3
    # conv window sizes
    conv_1_window_size = 11
    conv_2_window_size = 5
    conv_3_1_window_size = 3
    conv_3_2_window_size = 3
    conv_3_3_window_size = 3
    
    # pool stride sizes
    pool_1_strides = 2
    pool_2_strides = 2
    pool_3_strides = 2
    # conv stride sizes
    conv_1_strides = 4
    conv_2_strides = None #Default
    conv_3_1_strides = None #Default
    conv_3_2_strides = None #Default
    conv_3_3_strides = None #Default
    # compressed data size
    compressed_img_height = img_height/pool_1_window_size/pool_2_window_size
    compressed_img_width = img_width/pool_1_window_size/pool_2_window_size
    # nodes
    n_nodes_conv_layer_1 = 96
    n_nodes_conv_layer_2 = 256
    n_nodes_conv_layer_3_1 = 384
    n_nodes_conv_layer_3_2 = 384
    n_nodes_conv_layer_3_3 = 256
    n_nodes_fc_layer_4 = 4096
    n_nodes_fc_layer_5 = 4096
    # input changes for fully connected
    n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
    #
    with first_graph.as_default():
        # Input Layer
        first_convnet = input_data(shape=[None,img_width,img_height,color_channels], name='{}_input'.format(run_id)) # name should be different between models
        # Convolution - Pool Layer 1
        first_convnet = conv_2d(first_convnet, n_nodes_conv_layer_1, conv_1_window_size, strides=conv_1_strides, activation='relu')
        first_convnet = max_pool_2d(first_convnet, pool_1_window_size, strides=pool_1_strides)
        first_convnet = local_response_normalization(first_convnet)
        # Convolution - Pool Layer 2
        first_convnet = conv_2d(first_convnet, n_nodes_conv_layer_2, conv_2_window_size, activation='relu')
        first_convnet = max_pool_2d(first_convnet, pool_2_window_size, strides=pool_2_strides)
        first_convnet = local_response_normalization(first_convnet)
        # 3 Convolutions 1 Pool Layer 3
        first_convnet = conv_2d(first_convnet, n_nodes_conv_layer_3_1, conv_3_1_window_size, activation='relu')
        first_convnet = conv_2d(first_convnet, n_nodes_conv_layer_3_2, conv_3_2_window_size, activation='relu')
        first_convnet = conv_2d(first_convnet, n_nodes_conv_layer_3_3, conv_3_3_window_size, activation='relu')
        first_convnet = max_pool_2d(first_convnet, pool_3_window_size, strides=pool_3_strides)
        first_convnet = local_response_normalization(first_convnet)
        # Fully connected layer 4
        first_convnet = fully_connected(first_convnet, n_nodes_fc_layer_4, activation='tanh')
        first_convnet = dropout(first_convnet, 0.5) # 50% keep rate
        # Fully connected layer 4
        first_convnet = fully_connected(first_convnet, n_nodes_fc_layer_5, activation='tanh')
        first_convnet = dropout(first_convnet, 0.5) # 50% keep rate
        ###
        # Output layer
        first_convnet = fully_connected(first_convnet, n_classes, activation='softmax')
        first_convnet = regression(
            first_convnet, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id) # name should be different between models
            )
        # Set the model to follow this network
        # tensorboard_verbose = 0: Loss, Accuracy (Best Speed)
        tensorboard_log = MakeLogFile('')
        first_model = tflearn.DNN(first_convnet, tensorboard_dir=tensorboard_log)
    return first_model, first_graph

#### Model defining
## I have to set the network with different variable names or tensorflow has errors
def Convolutional_Model_Alexnet_second(img_height,img_width,color_channels, n_classes, run_id='', second_graph=tf.Graph()):
    # data size
    pixels = img_height*img_width
    batch_size = 64
    # pool window sizes
    pool_1_window_size = 3
    pool_2_window_size = 3
    pool_3_window_size = 3
    # conv window sizes
    conv_1_window_size = 11
    conv_2_window_size = 5
    conv_3_1_window_size = 3
    conv_3_2_window_size = 3
    conv_3_3_window_size = 3
    
    # pool stride sizes
    pool_1_strides = 2
    pool_2_strides = 2
    pool_3_strides = 2
    # conv stride sizes
    conv_1_strides = 4
    conv_2_strides = None #Default
    conv_3_1_strides = None #Default
    conv_3_2_strides = None #Default
    conv_3_3_strides = None #Default
    # compressed data size
    compressed_img_height = img_height/pool_1_window_size/pool_2_window_size
    compressed_img_width = img_width/pool_1_window_size/pool_2_window_size
    # nodes
    n_nodes_conv_layer_1 = 96
    n_nodes_conv_layer_2 = 256
    n_nodes_conv_layer_3_1 = 384
    n_nodes_conv_layer_3_2 = 384
    n_nodes_conv_layer_3_3 = 256
    n_nodes_fc_layer_4 = 4096
    n_nodes_fc_layer_5 = 4096
    # input changes for fully connected
    n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
    #
    with second_graph.as_default():
        # Input Layer
        second_convnet = input_data(shape=[None,img_width,img_height,color_channels], name='{}_input'.format(run_id)) # name should be different between models
        # Convolution - Pool Layer 1
        second_convnet = conv_2d(second_convnet, n_nodes_conv_layer_1, conv_1_window_size, strides=conv_1_strides, activation='relu')
        second_convnet = max_pool_2d(second_convnet, pool_1_window_size, strides=pool_1_strides)
        second_convnet = local_response_normalization(second_convnet)
        # Convolution - Pool Layer 2
        second_convnet = conv_2d(second_convnet, n_nodes_conv_layer_2, conv_2_window_size, activation='relu')
        second_convnet = max_pool_2d(second_convnet, pool_2_window_size, strides=pool_2_strides)
        second_convnet = local_response_normalization(second_convnet)
        # 3 Convolutions 1 Pool Layer 3
        second_convnet = conv_2d(second_convnet, n_nodes_conv_layer_3_1, conv_3_1_window_size, activation='relu')
        second_convnet = conv_2d(second_convnet, n_nodes_conv_layer_3_2, conv_3_2_window_size, activation='relu')
        second_convnet = conv_2d(second_convnet, n_nodes_conv_layer_3_3, conv_3_3_window_size, activation='relu')
        second_convnet = max_pool_2d(second_convnet, pool_3_window_size, strides=pool_3_strides)
        second_convnet = local_response_normalization(second_convnet)
        # Fully connected layer 4
        second_convnet = fully_connected(second_convnet, n_nodes_fc_layer_4, activation='tanh')
        second_convnet = dropout(second_convnet, 0.5) # 50% keep rate
        # Fully connected layer 4
        second_convnet = fully_connected(second_convnet, n_nodes_fc_layer_5, activation='tanh')
        second_convnet = dropout(second_convnet, 0.5) # 50% keep rate
        ###
        # Output layer
        second_convnet = fully_connected(second_convnet, n_classes, activation='softmax')
        second_convnet = regression(
            second_convnet, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id) # name should be different between models
            )
        # Set the model to follow this network
        # tensorboard_verbose = 0: Loss, Accuracy (Best Speed)
        tensorboard_log = MakeLogFile('')
        second_model = tflearn.DNN(second_convnet, tensorboard_dir=tensorboard_log)
    return second_model, second_graph

def Train(mode = 'grayscale_resize_1to10', which = 'first', first_graph=tf.Graph(), second_graph=tf.Graph()):
    # Train any amount of 10x epochs so it gets to at least 15 mins error #
    ## Data reorganize
    X,Y,test_x,test_y = LoadData(mode = mode, which=which)
    # X and test_x have the shape (sample_size, img_height, img_width, color_channels) from OpenCV.
    # Mine is Grayscale, so the shape is actually (sample_size, img_height, img_width)
    # Y and test_y are OneHot encoded so have the shape (sample_size, n_classes)
    img_height = len(X[0])
    img_width = len(X[0][0])
    if type(X[0][0][0])==type(numpy.array([])):
        color_channels = len(X[0][0][0])
    else:
        color_channels = 1
    # tensorflow needs the input to be width first and height second, backwards from what OpenCV outputs
    X = numpy.transpose(X, [0,2,1]) 
    test_x = numpy.transpose(test_x, [0,2,1])
    X = X.reshape([-1,img_width,img_height,color_channels]) 
    test_x = test_x.reshape([-1,img_width,img_height,color_channels])
    n_classes = len(Y[0])
    ##
    version = 1
    times_run = 1
    times_10x = 1
    batch_size = None
    model_name = 'Your_model_name_{}_{}class__v{}'.format(mode, which, version)
    run_id = '{}_run{}'.format(model_name,times_run)
    while os.path.exists(os.path.join(MakeLogFile(''), run_id)):
        times_run += 1
        run_id = '{}_run{}'.format(model_name,times_run)
    model_path = os.path.abspath(MakeModelPath('',server=True)+'/{0}/{0}.tfl'.format(run_id))
    model_dir = os.path.abspath(MakeModelPath('',server=True)+'/{0}'.format(run_id))
    if which == 'first':
        first_model, first_graph = Convolutional_Model_Alexnet_first(img_height,img_width,color_channels,n_classes=n_classes,run_id=run_id, first_graph=first_graph)
        with first_graph.as_default():
            first_model.fit(
                {'{}_input'.format(run_id): X},
                {'{}_targets'.format(run_id): Y},
                n_epoch=5,
                validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                show_metric=True,
                batch_size=batch_size,
                shuffle=False,
                snapshot_step=None,
                run_id=run_id
                )
            first_model.save(model_path)
    elif which == 'second':
        second_model, second_graph = Convolutional_Model_Alexnet_second(img_height,img_width,color_channels,n_classes=n_classes,run_id=run_id, second_graph=second_graph)
        with second_graph.as_default():
            second_model.fit(
                {'{}_input'.format(run_id): X},
                {'{}_targets'.format(run_id): Y},
                n_epoch=5,
                validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                show_metric=True,
                batch_size=batch_size,
                shuffle=False,
                snapshot_step=None,
                run_id=run_id
                )
            second_model.save(model_path)
    print_log_instructions()

def main():
    mode = 'grayscale_resize_1to10'
    first_graph = tf.Graph()
    second_graph = tf.Graph()
    Train(which='first', first_graph=first_graph)
    Train(which='second', second_graph=second_graph)

if __name__ == '__main__':
    main()
