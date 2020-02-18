# ComplexNet
The keras implementation of my undergraduate project"Human Pose 3D Estimation"

## Main idea:
  1. estimate the 2D joint position with stacked hourglass.
  2. Using the generated 2D joint coorperates to generate 3D pose
## Innovative parts:
  1. Unsupervised learning:
      It's not even necessary for you to feed this model 2D labeled data, just the 2D joints will get all net worked.
  2. 3D data FREE:
      You don't even need the 3D data to train your model, by firstly generate the 3D pose and 
      secondly generate the 3D pose by transformation and inverse transformation, then get MPJPE loss from these two 3D pose will
      eliminate the scarecity of 3D data.
  3. Flexible: 
      Someone might wonder that what if I got some 3D pose data? Congratulations! You can absolutely firstly 
      compute the 3D regression loss to get the model "warm start", that will help a lot.
  4. Overfitting avoided:
      By using discriminator to fit the normal distribution of normal human pose and geometrical constraints of human limb symmetry, some traits learned by the model can be 
      robust to multiple types of data.
 # process:
  This building is nowadays  Under Construction....
 
 
 Written by a bodybuilder, Yufei Zheng
