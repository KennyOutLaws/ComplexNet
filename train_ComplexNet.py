import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import glob
import h5py
import random
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
from keras.layers import Dense,Input,Activation,Lambda, Reshape,Flatten, concatenate, LeakyReLU,BatchNormalization,Dropout
from keras.models import  Model, load_model, Sequential
import keras.backend as K
import keras.layers as L
from functools import partial
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from  eval_functions import err_3dpe


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def kcs_layer(x):
    # implementation of the Kinematic Chain Space as described in the paper

    import tensorflow as tf

    # KCS matrix
    Ct = tf.constant([
          [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
          [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 1, 0],
          [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 , 0, 0, 0, 0,-1],
          [0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,-1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,-1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,-1, 0, 0]])

    C = tf.reshape(tf.tile(Ct, (tf.shape(x)[0], 1)), (-1, 16, 15))

    poses3 = tf.to_float(tf.reshape(x, [-1, 3, 16]))
    B = tf.matmul(poses3, C)
    Psi = tf.matmul(tf.transpose(B, perm=[0, 2, 1]), B)

    return Psi


def transformation_forward(pose_in):
    x = pose_in[:, 0:16]
    y = pose_in[:, 16:32]
    z = pose_in[:, 32:48]
    pose_cat = tf.concat([y,x,z], axis = 1)
   # pose_out = tf.squeeze(pose_cat)
    return pose_cat


def transformation_inverse(pose_in):
    y = pose_in[:, 0:16]
    x = pose_in[:, 16:32]
    z = pose_in[:, 32:48]
    pose_cat =  tf.concat([x,y,z], axis = 1)
   # pose_out = tf.squeeze(pose_cat)
    return pose_cat


def weighted_pose_2d_loss(y_true, y_pred):
    # the custom loss functions weights joints separately
    # it's possible to completely ignore joint detections by setting the respective entries to zero

    diff = tf.to_float(tf.abs(y_true - y_pred))

    # weighting the joints
    weights_t = tf.to_float(
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))

    weights = tf.tile(tf.reshape(weights_t, (1, 32)), (tf.shape(y_pred)[0], 1))

    tmp = tf.multiply(weights, diff)

    loss = tf.reduce_sum(tmp, axis=1) / 32

    return loss


def wasserstein_loss(y_true, y_pred):

    return K.mean(y_true * y_pred)


def mpjpe_pred1_2(y_true, y_pred):
   '''
    cnt, pjpe = 0, 0
    for i in range(y_pred.shape[0]):
        cnt += 1
        err = (((y_pred[i] - y_true[i])**2).sum(axis = 1)**0.5).mean()
        pjpe += err
    if cnt > 0:
        pjpe /= cnt
    return pjpe
'''
def mpjpe_pred1_2(y_true, y_pred):
    x1 = y_true[:, 0:16]
    x2 = y_pred[:, 0:16]
    y1 = y_true[:, 16:32]
    y2 = y_pred[:, 16:32]
    z1 = y_true[:, 0:16]
    z2 = y_pred[:, 16:32]
    return tf.reduce_mean(tf.sqrt(tf.square(x1 - x2)+tf.square(y1-y2)+tf.square(z1 - z2)))

def geometrical_constraint_loss(y_pred):
    x = y_pred[:, 0:16]
    y = y_pred[:, 16:32]
    z = y_pred[:, 32:48]
    # define the ratio of the arm and leg
    weights_ratio = [0.4, 0.6]

    # left limb 7->10->11->12, right limb 7->13->14->15
    left_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 7] - x[:, 10])+tf.square(y[:, 7] - y[:, 10])+tf.square(z[:, 7] - z[:, 10]))+tf.sqrt(
        tf.square(x[:, 10] - x[:, 11])+tf.square(y[:, 10] - y[:, 11])+tf.square(z[:, 10] - z[:, 11]))+tf.sqrt(
        tf.square(x[:, 11] - x[:, 12])+tf.square(y[:, 11] - y[:, 12])+tf.square(z[:, 11] - z[:, 12])))
    right_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 7] - x[:, 13])+tf.square(y[:, 7] - y[:, 13])+tf.square(z[:, 7] - z[:, 13]))+tf.sqrt(
        tf.square(x[:, 13] - x[:, 14])+tf.square(y[:, 13] - y[:, 14])+tf.square(z[:, 13] - z[:, 14]))+tf.sqrt(
        tf.square(x[:, 14] - x[:, 15])+tf.square(y[:, 14] - y[:, 15])+tf.square(z[:, 14] - z[:, 15])))
    left_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 6] - x[:, 3]) + tf.square(y[:, 6] - y[:, 3]) + tf.square(z[:, 6] - z[:, 3])) + tf.sqrt(
        tf.square(x[:, 3] - x[:, 4]) + tf.square(y[:, 3] - y[:, 4]) + tf.square(z[:, 3] - z[:, 4])) + tf.sqrt(
        tf.square(x[:, 4] - x[:, 5]) + tf.square(y[:, 4] - y[:, 5]) + tf.square(z[:, 4] - z[:, 5])))
    right_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 6] - x[:, 0]) + tf.square(y[:, 6] - y[:, 0]) + tf.square(z[:, 6] - z[:, 0])) + tf.sqrt(
        tf.square(x[:, 0] - x[:, 1]) + tf.square(y[:, 0] - y[:, 1]) + tf.square(z[:, 0] - z[:, 1])) + tf.sqrt(
        tf.square(x[:, 1] - x[:, 2]) + tf.square(y[:, 1] - y[:, 2]) + tf.square(z[:, 1] - z[:, 2])))
    loss = tf.reduce_mean(left_arm - right_arm) * weights_ratio[0] + tf.reduce_mean(left_leg - right_leg) * weights_ratio[1]
    return loss



def diy_loss(y_true, y_pred):
    return y_pred


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:n
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    # ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    # ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


net_name = 'ComplexNet'
path_h36m = './data/h36m/'
files = glob.glob(path_h36m+'S*'+'/StackedHourglass'+'/*.h5')
print('training ' + net_name)
print('loading training data...')
print('loading Human3.6M')

'''
poses_2d_list = []
for path in files:
    f = h5py.File(path, 'r')
    poses_2d_list.extend(f[u'poses'])
'''

poses_2d_list = h5py.File(files[0], 'r')['poses']

print('Done!')

print('loading test data...')
print('loading Human3.6M')
file_2d_eval = glob.glob(path_h36m+'S11'+'/StackedHourglass'+'/*.h5')
file_3d_eval = glob.glob(path_h36m+'S11'+'/MyPoses/3D_positions'+'/*.h5')

# Need modification
pose_2d_eval = np.array(h5py.File(file_2d_eval[0], 'r')['poses'])
pose_2d_eval[:, 16:32] = -pose_2d_eval[:, 16:32]
pose_3d_eval = np.array(h5py.File(file_3d_eval[0], 'r')['3D_positions']).T
pose_3d_eval = pose_3d_eval / 1000
poses_2d = np.array(poses_2d_list)
poses_2d[:, 16:32] = -poses_2d[:, 16:32]

num_joints = 16
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 32
TRAINING_RATIO = 5


# Lifting Network
pose_in_l = Input(shape=(2*num_joints,))
lift1 = Dense(1000)(pose_in_l)
lift1 = LeakyReLU()(lift1)

# Residual block added
lift21 = Dense(1000)(lift1)
lift21 = LeakyReLU()(lift21)
lift22 = Dense(1000)(lift21)
lift22 = L.add([lift1, lift22])
lift22 = LeakyReLU()(lift22)

lift31 = Dense(1000)(lift22)
lift31 = LeakyReLU()(lift31)
lift32 = Dense(1000)(lift31)
lift32 = L.add([lift32,lift22])
lift32 = LeakyReLU()(lift32)

lift41 = Dense(1000)(lift32)
lift41 = LeakyReLU()(lift41)
lift42 = Dense(1000)(lift41)
lift42 = L.add([lift42,lift32])
lift42 = LeakyReLU()(lift42)

lift5 = Dense(1000)(lift42)
lift5 = LeakyReLU()(lift5)
pose_out = Dense(3*num_joints)(lift5)

lifting_model = Model(inputs=pose_in_l, outputs=pose_out)
# Reprojection
pose_in_3D = Input((3*num_joints,))
reproj1 = Dense(1000)(pose_in_3D)
reproj1 = LeakyReLU()(reproj1)

## Residual Block


def residual_block_reproj(residual_input):
    residual2 = Dense(1000)(residual_input)
    residual2 = BatchNormalization()(residual2)
    residual2 = LeakyReLU()(residual2)
    residual2 = Dropout(0.4)(residual2)

    residual3 = Dense(1000)(residual2)
    residual3 = BatchNormalization()(residual3)
    residual3 = LeakyReLU()(residual3)
    residual3 = Dropout(0.4)(residual3)

    residual_add = L.add([residual_input,residual3])
    residual_out = LeakyReLU()(residual_add)
    return residual_out

#residual_block_reproj = Model(inputs=residual1,outputs = residual_out)


# Reprojection Module 1
reproj2_res = residual_block_reproj(reproj1)
reproj3_res = residual_block_reproj(reproj2_res)

# KCS path
psi = Lambda(kcs_layer)(pose_in_3D)
psi_vec = Flatten()(psi)
psi_vec = Dense(1000)(psi_vec)
psi_vec = LeakyReLU()(psi_vec)

d1_psi = Dense(1000)(psi_vec)
d1_psi = LeakyReLU()(d1_psi)
d2_psi = Dense(1000)(d1_psi)
d2_psi = L.add([psi_vec,d2_psi])
d2_psi = LeakyReLU()(d2_psi)

reproj_con = concatenate([reproj3_res, d2_psi])
reproj_last = Dense(100)(reproj_con)
reproj_last = LeakyReLU()(reproj_last)
reproj_out = Dense(2*num_joints)(reproj_last)

reprojection_model = Model(inputs = pose_in_3D, outputs=reproj_out)

# 2D Pose Discriminator
discriminator_in = Input((2*num_joints,))
discriminator1 = Dense(100)(discriminator_in)
discriminator1 = LeakyReLU()(discriminator1)

discriminator2 = Dense(100)(discriminator1)
discriminator2 = LeakyReLU()(discriminator2)
discriminator3 = Dense(100)(discriminator2)
discriminator3 = L.add([discriminator3,discriminator1])
discriminator3 = LeakyReLU()(discriminator3)

discriminator4 = Dense(100)(discriminator3)
discriminator4 = LeakyReLU()(discriminator4)
discriminator5 = Dense(100)(discriminator4)
discriminator5 = L.add([discriminator5,discriminator3])
discriminator5 = LeakyReLU()(discriminator5)

discriminator_out = Dense(1)(discriminator5)

discriminator = Model(inputs=discriminator_in, outputs=discriminator_out)

# Network Structure
pose_in_2D = Input((2*num_joints,))
pose_3d_1 = lifting_model(pose_in_2D)
# Geometrical symmetry regularization for 3d pose 1
loss_3d_geometric_1 = Lambda(geometrical_constraint_loss)(pose_3d_1)
random_transfer_3d = Lambda(transformation_forward)(pose_3d_1)
pose_2d_1 = reprojection_model(random_transfer_3d)
discriminator_for_generator = discriminator(pose_2d_1)
pose_3d_2 = lifting_model(pose_2d_1)
inverse_transfer_3d = Lambda(transformation_inverse)(pose_3d_2)
loss_3d = Lambda(mpjpe_pred1_2, arguments={'y_pred': random_transfer_3d})(inverse_transfer_3d)
# Geometrical symmetry regularization for 3d pose 2
loss_3d_geometric_2 = Lambda(geometrical_constraint_loss)(pose_3d_2)
pose_2d_2 = reprojection_model(inverse_transfer_3d)
loss_2d = Lambda(weighted_pose_2d_loss, arguments={'y_pred':pose_2d_2})(pose_2d_1)


generator = Model(inputs=[pose_in_2D], outputs = [pose_2d_1])


for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

adversarial_model = Model(inputs=[pose_in_2D], outputs=[discriminator_for_generator, loss_3d_geometric_1, loss_3d_geometric_2, loss_3d, loss_2d])
adversarial_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss, diy_loss, diy_loss, diy_loss, diy_loss])

# Now that the generator model is compiled, we can make the discriminator layers trainable
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False


real_samples = Input(shape=(2*num_joints, ))
generator_input_for_discriminator = Input(shape=(2*num_joints,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out = discriminator(averaged_samples)

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

partial_gp_loss.__name__ = 'gradient_penalty'

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
discriminator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)


for epoch in range(20):
    np.random.shuffle(poses_2d)
    print("Epoch: ", epoch)
    print("Number of batches: ",int(poses_2d.shape[0]//BATCH_SIZE))
    discriminator_loss = []
    adversarial_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in  range(int(poses_2d.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        noise_minibatches = poses_2d[i * minibatches_size:(i+1)*minibatches_size]
        rand_samples = random.sample(range(0, poses_2d.shape[0]), minibatches_size)
        discriminator_minibatches = poses_2d[rand_samples, :]
        for j in range(TRAINING_RATIO):
            pose_batch = discriminator_minibatches[j * BATCH_SIZE:(j+1) * BATCH_SIZE]
            noise = noise_minibatches[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
            discriminator_loss.append(discriminator_model.train_on_batch([pose_batch, noise], [positive_y, negative_y, dummy_y]))
            adversarial_loss.append(adversarial_model.train_on_batch(noise, [np.matlib.ones((BATCH_SIZE, 1)), np.matlib.zeros((BATCH_SIZE, 1)), np.matlib.zeros((BATCH_SIZE, 1)),np.matlib.zeros((BATCH_SIZE, 1))]))

    #   需要改造一下

        if i % 100 == 0 and i > 0:
            pred = lifting_model.predict(pose_2d_eval)
            # caculate training
            val = 0
            for p in range(200):
                val = val + 1000*err_3dpe(pose_3d_eval[p:p+1, :], pred[p:p+1, :])
                val = val/200
                #  loss_3d_geometric_1, loss_3d_geometric_2, loss_3d, loss_2d
                sys.stdout.write("\rIteration %d: 3d_error: %.3e, 3d_geometric_1,2: %.3e, %.3e, double_lifting_error: %.3e, reprojection_error: %.3e, disc_loss: %.3e "
                                  %  (i, val, adversarial_loss[-1][0],adversarial_loss[-1][1],adversarial_loss[-1][2], adversarial_loss[-1][3],discriminator_loss[-1][0]))
                sys.stdout.flush()
        # save model every 1000 iterations
        if i % 1000 ==0 and i > 0:
            lifting_model.save('models/tmp/lifting_'+ net_name+'.h5')
            generator.save('models/tmp/generator_' + net_name+'.h5')
            discriminator.save('models/tmp/discriminator_'+ net_name+'.h5')
            reprojection_model.save('models/tmp/reprojection_'+ net_name+'.h5')
            adversarial_model.save('models/tmp/adversarial_model_'+ net_name+'.h5')
        # decrease learning rate every 5 epochs
        if epoch % 5 ==0 and epoch >0:
            lrd = K.get_value(discriminator.optimizer.lr)
            lrd = lrd / 10
            K.set_value(discriminator.optimizer.lr, lrd)
            # set new learning rate for adversarial model
            lra = K.get_value(adversarial_model.optimizer.lr)
            lra = lra / 10
            K.set_value(adversarial_model.optimizer.lr, lra)
session.close()
