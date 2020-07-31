# Image-Matching
A demo using [SuperGlue and SuperPoint](https://github.com/magicleap/SuperGluePretrainedNetwork.git) to do the image matching task.



## image matching

### Overview

- Deeplearning Framework: PyTorch
- use `SuperPoint` to extract feature points and descriptors from the image
- use `SuperGlue` to do image matching
- `Matching` is a class that combine the above two classes
- use `Arg` class to make configurations

### Usage

- use `args=Arg()` to initialize configurations 
- Since the model has two mode to make more concise predicitions, if the scene is indoor, set scene_mode in Arg ‘indoor’. Similarly set the scene_mode    ‘outdoor’ if the scene is outdoor
- in class Arg, use `set_` method to set parameters
- To be clear, the anchor image should be set first in `Arg` by using `Arg.set_anchor(anchor_name)`, the image to test (if it can match the anchor image)  should be set using the method `Arg.set_match(match_name)`
- After setting the anchor image, you need not set it again unless you want to make another matching with a different anchor
- use `get_similarity(args,img_name)` to get the final score
- If the matching score is larger than 0.2, it is likely can be matched



### To-dos

- Improve the matching speed, now it takes 3s to do a matching

  





## pose recognition

Waiting...