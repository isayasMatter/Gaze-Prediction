# Eye gaze prediction OpenEDS 2020 Challenge

This repo contains the code I wrote for the OpenEDS 2020 Challenge. You can learn more about the challenge [here](https://research.fb.com/programs/openeds-2020-challenge).
The challenge had two tracks, one of which was the gaze prediction challenge. The challange's main goals were:

* to predict future gaze locations based on previously estimated eye gaze vectors
* to predict future gaze locations based on previously estimated gaze vectors while additionally leveraging spatio-temporal information encoded in sequence of previously recorded eye-images.

## My submission

I used Tensorflow with Keras to implement a recurrent convolutional neural network. The network consists of two parts: 1) a Resnet (various variants) based part that learn features from each eye image, and 2) a many-to-one recurrent neural network based temporal part that predicts gaze positions.

Although my submission was not the winning model :disappointed_relieved:,	 it scored way better than the baseline score and very close to the winning submission. Feel free to use the code as you see fit and shoot me an email if you have any questions about the code.
