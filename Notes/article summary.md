# Real-Time Segmentation of Non-Rigid Surgical Tools based on Deep Learning and Tracking

https://www.researchgate.net/publication/305770331_Real-Time_Segmentation_of_Non-Rigid_Surgical_Tools_based_on_Deep_Learning_and_Tracking

## Applications

- Increase the context-awareness of surgeons in the operating room.
- Provide accurate real-time information about the surgical tools.
- Mosaicking.
- Visual servoing.
- Skills assessment.


## Pros

- No need to modify the current workflow or propose alternative exotic instruments.
- works with non-rigid tools with unknown geometry and kinematics.

## Challenges

### Segmentation

- Specular reflections.
- Changing lighting conditions.
- Shadows.
- Motions blur.
- Scene complexity.
- Motion of organs.
- Occlusions (body fluids and smoke).
- Poor resolution.
- Non-Rigid instruments.

### Others

- Real-time (~30fps).

## Method

### CNN (*Convolutional Neural Network*)

#### Benefits

- No need for trial and error to hand-craft features, as features are automatically extracted during the network training phase.
- Automatic feature selection does not negatively affect the segmentation quality.
- Trained on large general purpose datasets and then fine-tuned on few domain-specific images => not tool
dependent.

### FCN (*Fully Convolutional Networks*)

- Tailored to perform semantic labelling rather than classification (≠CNN).

How to convert from classification to segmentation:

- FC (*Fully Connected*) layers replaced with convolutions (spatial information is preserved).
- Upsampling filters (*deconvolution layers*) => arbitrary-sized input, produce a labelled output of equivalent dimensions.

#### Network used (FCN-8s)

- pre-trained on the PASCAL-context 59-class (60 including background) dataset.
- change the number of
outputs to just 2 in the scoring and upsampling layers.
- final per-pixel scores provided by the FCN are normalized and calculated via *argmax* to obtain per-pixel labels.
- Optimizer: *Stochastic Gradient Descent* (SGD).
- Triangular *Learning rate* (LR) through time. Small value to not alter the behavior. LR boundaries: [1e-13, 1e-10], momentum: 0.99, weight decay: 0.0005.

### Real-Time pipeline

100ms for 500x500 RGB image, so not real-time.

#### Asynchronous pipeline

- FCN computes segmentation each ~100ms.
- Extract tracking points for FCN segmented only.
- Tracking is done on greyscale image.
- Max 4000 tracking points.
- _GoodFeaturesToTrack_ extractor (openCV).
- Affine interpolation between the positions of points for current image and last FCN-segmented image. Computed with a RANSAC approach (_estimateRigidTransform_, openCV implementation).
- Apply affine transformation to the mask of the last FCN-segmented image.

## Experiments and results

_TODO_
