Tensorflow Implementation

Caffe Implementation provided by authors at: [https://github.com/tinghuiz/appearance-flow] (https://github.com/tinghuiz/appearance-flow)




(The following description is taken as is from the caffe repo)

# [View Synthesis by Appearance Flow](https://arxiv.org/abs/1605.03557)
[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Shubham Tulsiani](https://people.eecs.berkeley.edu/~shubhtuls/), [Weilun Sun](http://sunweilun.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), and [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/), ECCV 2016.

## Overview

The paper addresses the problem of novel view synthesis: given an input image, synthesizing new images of the same object or scene observed from arbitrary viewpoints. We approach this as a learning task but, critically, instead of learning to synthesize pixels from scratch, we learn to copy them from the input image. Our approach exploits the observation that the visual appearance of different views of the same instance is highly correlated, and such correlation could be explicitly learned by training a convolutional neural network (CNN) to predict **appearance flows** – 2-D coordinate vectors specifying which pixels in the input view could be used to reconstruct the target view. Furthermore, the proposed framework easily generalizes to multiple input views by learning how to optimally combine single-view predictions. 

#### Single-view network architecture:
<img src='webpage/single_net.png' width=500>

#### Multi-view network architecture:
<img src='webpage/multi_net.png' width=500>

Link to the [[Paper]](https://arxiv.org/abs/1605.03557) [[Poster]](https://people.eecs.berkeley.edu/~tinghuiz/projects/appearanceFlow/poster.pdf)

Please contact Tinghui Zhou (tinghuiz@berkeley.edu) if you have any questions.




## Sample ShapeNet Results on Single-view 3D object rotation
The input view is marked with green bounding boxes. All the other views are synthesized by our single-view object rotation network.

<img src='webpage/sample_results/car_single/01.gif' width="160"/> <img src='webpage/sample_results/car_single/02.gif' width="160"/> <img src='webpage/sample_results/car_single/03.gif' width="160"/> <img src='webpage/sample_results/car_single/04.gif' width="160"/> <img src='webpage/sample_results/car_single/05.gif' width="160"/>

## Sample KITTI Results on 3D Scene Fly-through
The task is to synthesize a fly-through effect for the 3D scene given only two input views (marked in green and red bounding boxes). All the intermediate frames are synthesized.

<img src='webpage/sample_results/kitti/01.gif' width="210" height="150"/> <img src='webpage/sample_results/kitti/02.gif' width="210" height="150"/> <img src='webpage/sample_results/kitti/03.gif' width="210" height="150"/> <img src='webpage/sample_results/kitti/04.gif' width="210" height="150"/>


