# TFlearn_Alexnet_video_analysis
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
