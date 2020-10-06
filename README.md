# TFlearn Alexnet video analysis with multiple networks
An implementation of Alexnet using OpenCV and TFlearn for a video analysis project I've been working on.

I had to train several models in a sequence, and what I found is that for loops with graphs don't work, and every model and network needs their own unique variables instead of replaceable ones.

For reference I used these:

Support for multiple DNN?:
https://github.com/tflearn/tflearn/issues/381

What happens if I used the for loop was the same issue as below 

Graph serialization error for LSTMs:
https://github.com/tflearn/tflearn/issues/605

In my code I write how I had to assemble 2 models with the same shape in tflearn
I had to repeat the same code with different variable names so that tensorflow didn't have any errors

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


I ran the programs in the order:

* __VideoProcessing.py__ : Reads video into frames jpg
* __Frame_to_vector.py__ : Reads the jpg into a vector with classification
* __tflearn_Alexnet.py__ : Runs the Alexnet model


For the video images I used OpenCV, and I installed as follows:

## OpenCV

### Ubuntu

UBUNTU OPENCV PYTHON3.6

http://www.python36.com/how-to-install-opencv340-on-ubuntu1604/

Ffmpeg dependency

http://ubuntuhandbook.org/index.php/2016/09/install-ffmpeg-3-1-ubuntu-16-04-ppa/

For python 3.6 and 2.7:

```
cmake -DCMAKE_BUILD_TYPE=Release \
    -D WITH_FFMPEG=ON \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
    -D OPENCV_ENABLE_NONFREE=True ..
```

Build but don’t install in a separate folder so that I get the python 3.5 module binding

```
cmake -D CMAKE_BUILD_TYPE=Release \
    -D WITH_FFMPEG=ON \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3.5 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
    -D OPENCV_ENABLE_NONFREE=True ..
```

### MacOSX

http://blog.jiashen.me/2014/12/23/build-opencv-3-on-mac-os-x-with-python-3-and-ffmpeg-support/

```
brew install ffmpeg
brew install cmake pkg-config
brew install jpeg libpng libtiff openexr
brew install eigen tbb
```

```
cmake -D CMAKE_BUILD_TYPE=Release \
    -D WITH_FFMPEG=ON \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BUILD_opencv_xfeatures2d=OFF \
    -D OPENCV_ENABLE_NONFREE=True ..
```

make -j4
sudo make install

