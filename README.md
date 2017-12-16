Tensorflow Implementation (Work in Progress)

Notes: (Differences from published paper)
-Fully convolutional for KITTI (Please see discussion
:[https://github.com/tinghuiz/appearance-flow/issues/7] )
-Relative Transforms (Please see
:[https://github.com/tinghuiz/appearance-flow/issues/8] )

References:
-bilinear sampling code integrated from :
https://github.com/iwyoo/tf-bilinear_sampler
-Deconv layers weights initialised with Bilinear Sampling weights similar to
popular FCN archiecture implementations

TODO:
-tweak code for multi-view prediction
-Integrate code for shapenet training (currently code works for KITTI training)


Contributions and suggestions welcome!

Caffe Implementation provided by authors at: [https://github.com/tinghuiz/appearance-flow] 




(The following description is taken as is from the caffe repo)

# [View Synthesis by Appearance Flow](https://arxiv.org/abs/1605.03557)
[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Shubham Tulsiani](https://people.eecs.berkeley.edu/~shubhtuls/), [Weilun Sun](http://sunweilun.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), and [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/), ECCV 2016.

## Overview

The paper addresses the problem of novel view synthesis: given an input image, synthesizing new images of the same object or scene observed from arbitrary viewpoints. We approach this as a learning task but, critically, instead of learning to synthesize pixels from scratch, we learn to copy them from the input image. Our approach exploits the observation that the visual appearance of different views of the same instance is highly correlated, and such correlation could be explicitly learned by training a convolutional neural network (CNN) to predict **appearance flows** – 2-D coordinate vectors specifying which pixels in the input view could be used to reconstruct the target view. Furthermore, the proposed framework easily generalizes to multiple input views by learning how to optimally combine single-view predictions. 

