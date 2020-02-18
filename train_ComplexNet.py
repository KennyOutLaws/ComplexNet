import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Dense,Input,Activation,Lambda, Reshape,Flatten, concatenate, LeakyReLU,BatchNormalization,Dropout
from keras.models import  Model, load_model, Sequential
import keras.backend as K
import keras.layers as L

from keras.layers.merge import _Merge
from keras.optimizers import Adam

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
    pose_cat = tf.concat(1,[y,x,z])
    pose_out = tf.squeeze(pose_cat)
    return pose_out


def transformation_inverse(pose_in):
    y = pose_in[:, 0:16]
    x = pose_in[:, 16:32]
    z = pose_in[:, 32:48]
    pose_cat =  tf.concat(1,[x,y,z])
    pose_out = tf.squeeze(pose_cat)
    return pose_out


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


def mpjpe_pred1_2(y_pred, y_true):
    cnt, pjpe = 0, 0
    for i in range(y_pred.shape[0]):
        cnt += 1
        err = (((y_pred[i] - y_true[i])**2).sum(axis = 1)**0.5).mean()
        pjpe += err
    if cnt > 0:
        pjpe /= cnt
    return pjpe


def geometrical_constraint_loss(y_pred):
    x = y_pred[:, 0:16]
    y = y_pred[:, 16:32]
    z = y_pred[:, 32:48]
    # define the ratio of the arm and leg
    weights_ratio = [0.4, 0.6]

    # left limb 7->10->11->12, right limb 7->13->14->15
    left_arm = tf.reduce_mean(tf.sqrt(tf.square(x[:, 7] - x[:, 10])+tf.square(y[:, 7] - y[:, 10])+tf.square(z[:, 7] - z[:, 10]))+tf.sqrt(tf.square(x[:, 10] - x[:, 11])+tf.square(y[:, 10] - y[:, 11])+tf.square(z[:, 10] - z[:, 11]))+tf.sqrt(tf.square(x[:, 11] - x[:, 12])+tf.square(y[:, 11] - y[:, 12])+tf.square(z[:, 11] - z[:, 12])), axis = 1)
    right_arm = tf.reduce_mean(tf.sqrt(tf.square(x[:, 7] - x[:, 13])+tf.square(y[:, 7] - y[:, 13])+tf.square(z[:, 7] - z[:, 13]))+tf.sqrt(tf.square(x[:, 13] - x[:, 14])+tf.square(y[:, 13] - y[:, 14])+tf.square(z[:, 13] - z[:, 14]))+tf.sqrt(tf.square(x[:, 14] - x[:, 15])+tf.square(y[:, 14] - y[:, 15])+tf.square(z[:, 14] - z[:, 15])), axis = 1)
    left_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 6] - x[:, 3]) + tf.square(y[:, 6] - y[:, 3]) + tf.square(z[:, 6] - z[:, 3])) + tf.sqrt(
        tf.square(x[:, 3] - x[:, 4]) + tf.square(y[:, 3] - y[:, 4]) + tf.square(z[:, 3] - z[:, 4])) + tf.sqrt(
        tf.square(x[:, 4] - x[:, 5]) + tf.square(y[:, 4] - y[:, 5]) + tf.square(z[:, 4] - z[:, 5])), axis=1)
    right_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 6] - x[:, 0]) + tf.square(y[:, 6] - y[:, 0]) + tf.square(z[:, 6] - z[:, 0])) + tf.sqrt(
        tf.square(x[:, 0] - x[:, 1]) + tf.square(y[:, 0] - y[:, 1]) + tf.square(z[:, 0] - z[:, 1])) + tf.sqrt(
        tf.square(x[:, 1] - x[:, 2]) + tf.square(y[:, 1] - y[:, 2]) + tf.square(z[:, 1] - z[:, 2])), axis=1)
    loss = tf.reduce_mean(left_arm - right_arm) * weights_ratio[0] + tf.reduce_mean(left_leg - right_leg) * weights_ratio[1]
    return loss


num_joints = 16
# Lifting Network
pose_in = Input(shape=(2*num_joints,))
lift1 = Dense(1000)(pose_in)
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

lifting_model = Model(inputs=pose_in,outputs=pose_out)
# Reprojection
pose_in_3D = Input((3*num_joints,))
reproj1 = Dense(1000)(pose_in_3D)
reproj1 = LeakyReLU()(reproj1)

## Residual Block
residual1 = Input((1000,))
residual2 = Dense(1000)(residual1)
residual2 = BatchNormalization()(residual2)
residual2 = LeakyReLU()(residual2)
residual2 = Dropout(0.4)(residual2)

residual3 = Dense(1000)(residual2)
residual3 = BatchNormalization()(residual3)
residual3 = LeakyReLU()(residual3)
residual3 = Dropout(0.4)(residual3)

residual_add = L.add([residual1,residual3])
residual_out = LeakyReLU()(residual_add)

residual_block_reproj = Model(inputs=residual1,outputs = residual_out)


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
pose_in = Input((2*num_joints,))
pose_3d_1 = lifting_model(pose_in)
# Geometrical symmetry regularization for 3d pose 1
loss_3d_geometric_1 = Lambda(geometrical_constraint_loss)(pose_3d_1)
random_transfer_3d = Lambda(transformation_forward)(pose_3d_1)
pose_2d_1 = reprojection_model(random_transfer_3d)
discriminator_result = discriminator(pose_2d_1)
pose_3d_2 = lifting_model(pose_2d_1)
inverse_transfer_3d = Lambda(transformation_inverse)(pose_3d_2)
loss_3d = Lambda(mpjpe_pred1_2)([inverse_transfer_3d, random_transfer_3d])
# Geometrical symmetry regularization for 3d pose 2
loss_3d_geometric_2 = Lambda(geometrical_constraint_loss)(pose_3d_2)
pose_2d_2 = reprojection_model(inverse_transfer_3d)
loss_2d = Lambda(weighted_pose_2d_loss)([pose_2d_1,pose_2d_2])


