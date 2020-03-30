###Brand New Network
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import glob
import h5py
import random
import numpy as np
import pickle
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



def reprojection_layer(x):
    # reprojection layer as described in the paper

    x = tf.to_float(x)

    pose3 = tf.reshape(tf.slice(x, [0, 0], [-1, 48]), [-1, 3, 16])

    m = tf.reshape(tf.slice(x, [0, 48], [-1, 6]), [-1, 2, 3])

    pose2_rec = tf.reshape(tf.matmul(m, pose3), [-1, 32])

    return pose2_rec


def prepare_training_data():
    # poses_2d 即为所放训练的数据集，相差的scale会被预测出来的camera net自动补偿
    path_H36M = './data/h36m/'
    actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
               'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
    subjects = [1,5,6,7,8]
    training_data = []
    for sub in subjects:
        for action in actions:
            files = glob.glob(path_H36M + 'S' + str(sub) + '/StackedHourglass/' + action + '*.h5')
            for file in files:
                f = h5py.File(file, 'r')
                training_data.extend(list(f[u'poses']))

    poses_2d = np.zeros((len(training_data), 32))

    for p_idx in range(len(training_data)):
        p1 = training_data[p_idx]
        # reshape to my joint representation
        pose_2d = p1[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :].T

        mean_x = np.mean(pose_2d[0, :])
        pose_2d[0, :] = pose_2d[0, :] - mean_x

        mean_y = np.mean(pose_2d[1, :])
        pose_2d[1, :] = pose_2d[1, :] - mean_y

        pose_2d = np.hstack((pose_2d[0, :], pose_2d[1, :]))
        poses_2d[p_idx, :] = pose_2d / np.std(pose_2d)
    return poses_2d


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


def random_2d_projection(pose_in_3d):
    # todo 有待测验
    theta = tf.random.uniform((1,1),minval=-2*np.pi,maxval=2*np.pi,dtype=tf.float32)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
    # 2D Projection
    x = pose_in_3d[:,0:16]
    y = pose_in_3d[:,16:32]
    z = pose_in_3d[:,32:48]
    new_x = tf.multiply(x,cos_theta) + tf.multiply(z,sin_theta)
    pose_2d = tf.concat([new_x,y],axis=1)
    return pose_2d


def calculate_geometrical_loss(pose_in_3d):
    x  = pose_in_3d[:,0:16]
    z = pose_in_3d[:,32:48]
    a0 = z[:,8] - z[:,7]
    b0 = x[:,8] - x[:,7]
    n0 = tf.sqrt(a0*a0+b0*b0)
    a1 = z[:,10] - z[:,13]
    b1 = x[:,10] - x[:,13]
    n1 = tf.sqrt(a1*a1+b1*b1)
    sin_angle =  (a0*b1 - a1*b0) / (n0*n1)
    return K.relu(-sin_angle)

def weighted_pose_2d_loss(y_true, y_pred):
    # the custom loss functions weights joints separately
    # it's possible to completely ignore joint detections by setting the respective entries to zero

    diff = tf.to_float(tf.abs(y_true - y_pred))

    # weighting the joints
    weights_t = tf.to_float(np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))

    weights = tf.tile(tf.reshape(weights_t, (1, 32)), (tf.shape(y_pred)[0], 1))

    tmp = tf.multiply(weights, diff)

    loss = tf.reduce_sum(tmp, axis=1) / 32

    return loss


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def cam_loss(y_true, y_pred):
    # loss function to enforce a weak perspective camera as described in the paper

    m = tf.reshape(y_pred, [-1, 2, 3])

    m_sq = tf.matmul(m, tf.transpose(m, perm=[0, 2, 1]))

    loss_mat = tf.reshape((2 / tf.trace(m_sq)), [-1, 1, 1])*m_sq - tf.eye(2)

    loss = tf.reduce_sum(tf.abs(loss_mat), axis=[1, 2])

    return loss

def geometrical_constraint_loss(y_pred):
    x = y_pred[:, 0:16]
    y = y_pred[:, 16:32]
    z = y_pred[:, 32:48]
    # define the ratio of the arm and leg
    weights_ratio = [0.4, 0.6]

    # left limb 7->10->11->12, right limb 7->13->14->15
    left_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 13] - x[:, 14])+tf.square(y[:, 13] - y[:, 14])+tf.square(z[:, 13] - z[:, 14]))+tf.sqrt(
        tf.square(x[:, 14] - x[:, 15])+tf.square(y[:, 14] - y[:, 15])+tf.square(z[:, 14] - z[:, 15])))
    right_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 10] - x[:, 11])+tf.square(y[:, 10] - y[:, 11])+tf.square(z[:, 10] - z[:, 11]))+tf.sqrt(
        tf.square(x[:, 11] - x[:, 12])+tf.square(y[:, 11] - y[:, 12])+tf.square(z[:, 11] - z[:, 12])))
    left_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 0] - x[:, 1]) + tf.square(y[:, 0] - y[:, 1]) + tf.square(z[:, 0] - z[:, 1])) + tf.sqrt(
        tf.square(x[:, 1] - x[:, 2]) + tf.square(y[:, 1] - y[:, 2]) + tf.square(z[:, 1] - z[:, 2])))
    right_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 3] - x[:, 4]) + tf.square(y[:, 3] - y[:, 4]) + tf.square(z[:, 3] - z[:, 4])) + tf.sqrt(
        tf.square(x[:, 4] - x[:, 5]) + tf.square(y[:, 4] - y[:, 5]) + tf.square(z[:, 4] - z[:, 5])))
    loss = tf.reduce_mean(tf.abs(left_arm - right_arm)) * weights_ratio[0] + tf.reduce_mean(tf.abs(left_leg - right_leg)) * weights_ratio[1]
    return tf.abs(loss)

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
poses_2d = prepare_training_data()
num_joints = 16
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 32
TRAINING_RATIO = 5

# Lifting Network
pose_in_l = Input(shape=(2*num_joints,))
lift1 = Dense(1000)(pose_in_l)
lift1 = LeakyReLU()(lift1)

# Shared residual block..Camera estimation net + 2D->3D lifting net
lift21 = Dense(1000)(lift1)
lift21 = LeakyReLU()(lift21)
lift22 = Dense(1000)(lift21)
lift22 = L.add([lift1, lift22])
lift22 = LeakyReLU()(lift22)

lift31 = Dense(1000)(lift22)
lift31 = LeakyReLU()(lift31)
lift32 = Dense(1000)(lift31)
lift32 = L.add([lift22,lift32])
lift32 = LeakyReLU()(lift32)

lift41 = Dense(1000)(lift32)
lift41 = LeakyReLU()(lift41)
lift42 = Dense(1000)(lift41)
lift42 = L.add([lift32,lift42])
lift42 = LeakyReLU()(lift42)

lift5 = Dense(1000)(lift42)
lift5 = LeakyReLU()(lift5)
pose_out = Dense(3*num_joints)(lift5)

lifting_model = Model(inputs=pose_in_l, outputs=pose_out)

# camera net
lc11 = Dense(1000)(lift22)
lc11 = LeakyReLU()(lc11)
lc12 = Dense(1000)(lc11)
lc12 = L.add([lift22,lc12])
lc12 = LeakyReLU()(lc12)

lc21 = Dense(1000)(lc12)
lc21 = LeakyReLU()(lc21)
lc22 = Dense(1000)(lc21)
lc22 = L.add([lc12,lc22])
lc22 = LeakyReLU()(lc22)
cam_out = Dense(6)(lc22)

# combine 3D pose and camera estimation
# It is decomposed later in the reprojection layer..
concat_3d_cam = concatenate([pose_out,cam_out])

# connect the reprojection layer
rec_pose = Lambda(reprojection_layer)(concat_3d_cam)


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
cam_net = Model(inputs=pose_in_l,outputs=cam_out)
rep_net = Model(inputs=pose_in_l,outputs=rec_pose)
generator = Model(inputs=pose_in_l,outputs=pose_out)
discriminator = Model(inputs=discriminator_in, outputs=discriminator_out)

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

generator_input = Input(shape=(2*num_joints,))
generator_layers = generator(generator_input)
loss_geometric = Lambda(geometrical_constraint_loss)(generator_layers)
loss_3d_heuristic = Lambda(calculate_geometrical_loss)(generator_layers)
random_projection_2d_pose = Lambda(random_2d_projection)(generator_layers)
discriminator_for_generator = discriminator(random_projection_2d_pose)
cam_net_layers_for_generator = cam_net(generator_input)
rep_net_layers_for_generator = rep_net(generator_input)

adversarial_model = Model(inputs = [generator_input],outputs=[discriminator_for_generator,rep_net_layers_for_generator,cam_net_layers_for_generator,
                                                              loss_geometric,loss_3d_heuristic])
adversarial_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss, weighted_pose_2d_loss, cam_loss,diy_loss,diy_loss], loss_weights=[1, 1, 1,0.4,0.4])

# Now that the generator model is compiled, we can make the discriminator layers trainable
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False


real_samples = Input(shape=(2*num_joints, ))
generator_input_for_discriminator = Input(shape=(2*num_joints,))
# TODO 感觉可能有点问题
generated_samples_for_discriminator = Lambda(random_2d_projection)(generator(generator_input_for_discriminator))
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
    print("Number of batches: ", int(poses_2d.shape[0]//BATCH_SIZE))
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
            adversarial_loss.append(adversarial_model.train_on_batch(noise, [np.ones((BATCH_SIZE, 1)), noise, np.zeros((BATCH_SIZE, 1)),np.zeros((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))]))
    #  TODO

        if i % 100 == 0 and i > 0:
                #  loss_3d_geometric_1, loss_3d_geometric_2, loss_3d, loss_2d
            sys.stdout.write("\rIteration %d: wasserstain_loss: %.3e, rep_err: %.3e,cam_err:%.3e,geometric_loss:%.3e, heuristic_loss: %.3e\n"
                              %  (i, adversarial_loss[-1][0],adversarial_loss[-1][1],adversarial_loss[-1][2],adversarial_loss[-1][3], adversarial_loss[-1][4]))

            try:
                with open("logs/log_" + net_name + ".txt", "a") as logfile:
                    logfile.write("\rEpoch%d: Iteration %d: wasserstain_loss: %.3e, rep_err: %.3e,cam_err:%.3e,geometric_loss:%.3e, heuristic_loss: %.3e\n"
                              %  (epoch, i,adversarial_loss[-1][0],adversarial_loss[-1][1],adversarial_loss[-1][2],adversarial_loss[-1][3], adversarial_loss[-1][4]))
            except:
                print('error while writing logfile')
            sys.stdout.flush()
        # save model every 1000 iterations
        if i % 1000 ==0 and i > 0:
            lifting_model.save('models/tmp/lifting_' + net_name+'.h5')
        # decrease learning rate every 5 epochs
        if epoch % 5 ==0 and epoch >0:
            lrd = K.get_value(discriminator_model.optimizer.lr)
            lrd = lrd / 10
            K.set_value(discriminator_model.optimizer.lr, lrd)
            # set new learning rate for adversarial model
            lra = K.get_value(adversarial_model.optimizer.lr)
            lra = lra / 10
            K.set_value(adversarial_model.optimizer.lr, lra)
session.close()