import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

def normalize_3d(pose):
    # 3Dモデルの正規化
    # hip(0)と各関節点の距離の平均値が1になるようにスケール
    xs = pose.T[0::3] - pose.T[0]  # 17
    ys = pose.T[1::3] - pose.T[1]  # 17
    zs = pose.T[2::3] - pose.T[2]  # 17
    ls = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2 + zs[1:] ** 2)  # 原点からの距離 17个关节点到中心点的距离
    scale = ls.mean(axis=0)  # 距离的平均
    pose = pose.T / scale  # 缩放 使距离的平均值为1
    # hip(0)が原点になるようにシフト
    pose[0::3] -= pose[0].copy()
    pose[1::3] -= pose[1].copy()
    pose[2::3] -= pose[2].copy()
    return pose.T, scale


def normalize_2d(pose):
    # 2DPoseの正規化
    # hip(0)と各関節点の距離の平均値が1になるようにスケール
    xs = pose.T[0::2] - pose.T[0]
    ys = pose.T[1::2] - pose.T[1]
    pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
    # hip(0)が原点になるようにシフト
    mu_x = pose[0].copy()
    mu_y = pose[1].copy()
    pose[0::2] -= mu_x
    pose[1::2] -= mu_y
    return pose.T


def list_to_array(list_data):
    a = list_data[0]
    for i in range(1,len(list_data)):
        a = np.concatenate([a, list_data[i]])
    return a


def prepare_training_data():

    H36M_NAMES = [''] * 32
    H36M_NAMES[0] = 'Hip'
    H36M_NAMES[1] = 'RHip'
    H36M_NAMES[2] = 'RKnee'
    H36M_NAMES[3] = 'RFoot'
    H36M_NAMES[6] = 'LHip'
    H36M_NAMES[7] = 'LKnee'
    H36M_NAMES[8] = 'LFoot'
    H36M_NAMES[12] = 'Spine'  # 脊椎
    H36M_NAMES[13] = 'Thorax'  # 胸部
    H36M_NAMES[14] = 'Neck/Nose'
    H36M_NAMES[15] = 'Head'
    H36M_NAMES[17] = 'LShoulder'
    H36M_NAMES[18] = 'LElbow'  # 肘
    H36M_NAMES[19] = 'LWrist'  # 手腕
    H36M_NAMES[25] = 'RShoulder'
    H36M_NAMES[26] = 'RElbow'
    H36M_NAMES[27] = 'RWrist'

    if not os.path.exists('data/h36m/sh_detect_2d.pkl'):
        print('Downloading detected 2D points by Stacked Hourglass.')
        os.system('wget --no-check-certificate "https://onedriv' + \
                  'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                  'D60FE71FF90FD%2118619&authkey=AMBf6RPcWQgjsh0" -O ' + \
                  'data/h36m/sh_detect_2d.pkl')




    with open('data/h36m/sh_detect_2d.pkl', 'rb') as f:
        p2d_sh = pickle.load(f)
    if not os.path.exists('data/h36m/points_3d.pkl'):
        print('Downloading 3D points in Human3.6M dataset.')
        os.system('wget --no-check-certificate "https://onedriv' + \
            'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
            'D60FE71FF90FD%2118616&authkey=AFIfEB6VYEZnhlE" -O ' + \
            'data/h36m/points_3d.pkl')
    with open('data/h36m/points_3d.pkl', 'rb') as f:
        p3d = pickle.load(f)

    training_data_2d = []
    eval_data_2d = []
    training_data_3d = []
    eval_data_3d = []
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    for subject in subjects:
        for action in p2d_sh[subject]:
            i = 0
            for camera_name in p2d_sh[subject][action]:
                # 对每一个action里面的视角，我们取后400个
                training_data_2d.append(p2d_sh[subject][action][camera_name])
                if i==0:
                    eval_data_2d.append(p2d_sh[subject][action][camera_name][-400:])
                    i+=1

        for action in p3d[subject]:
            training_data_3d.append(p3d[subject][action][:-400])
            eval_data_3d.append(p3d[subject][action][-400:])

    dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
    dim_to_use_y = dim_to_use_x + 1
    dim_to_use_z = dim_to_use_x + 2
    dim_to_use = np.array(
        [dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten()

    training_data_2d = list_to_array(training_data_2d)
    training_data_3d = list_to_array(training_data_3d)
    eval_data_2d = list_to_array(eval_data_2d)
    eval_data_3d = list_to_array(eval_data_3d)
    training_data_3d = training_data_3d[:, dim_to_use]
    eval_data_3d = eval_data_3d[:, dim_to_use]


    training_data_2d, eval_data_2d = normalize_2d(training_data_2d), normalize_2d(eval_data_2d)
    training_data_3d, _ = normalize_3d(training_data_3d)
    eval_data_3d, _ = normalize_3d(eval_data_3d)
    # 研究normalize中transpose的问题，具体可以从numpy数组取一个维度进行试验...
    return training_data_2d, training_data_3d, eval_data_2d, eval_data_3d

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])



def random_2d_projection(pose_in_3d):
    # todo 有待测验
    theta = tf.random.uniform((1,1),minval=-2*np.pi,maxval=2*np.pi,dtype=tf.float32)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
    # 2D Projection
    x = pose_in_3d[:,0::3]
    y = pose_in_3d[:,1::3]
    z = pose_in_3d[:,2::3]
    new_x = tf.multiply(x,cos_theta) + tf.multiply(z,sin_theta)
    pose_2d = tf.concat([new_x,y],axis=1)
    return pose_2d


def recovered_2d_projection(pose_in_3d):
    #todo 只是简单的进行x，y的截取
    x = pose_in_3d[:,0::3]
    y = pose_in_3d[:,1::3]
    return tf.concat([x,y],axis = 1)


def calculate_geometrical_loss(pose_in_3d):
    x  = pose_in_3d[:,0::3]
    z = pose_in_3d[:,2::3]
    a0 = z[:,9] - z[:,8]
    b0 = x[:,9] - x[:,8]
    n0 = tf.sqrt(a0*a0+b0*b0)
    a1 = z[:,14] - z[:,11]
    b1 = x[:,14] - x[:,11]
    n1 = tf.sqrt(a1*a1+b1*b1)
    sin_angle =  (a0*b1 - a1*b0) / (n0*n1)
    return K.relu(-sin_angle)


def transformation_forward(pose_in):
    x = pose_in[:, 0::3]
    y = pose_in[:, 1::3]
    z = pose_in[:, 2::3]
    pose_cat = tf.concat([y,x,z], axis = 1)
   # pose_out = tf.squeeze(pose_cat)
    return pose_cat


def transformation_inverse(pose_in):
    # 需要更改
    y = pose_in[:, 0::3]
    x = pose_in[:, 1::3]
    z = pose_in[:, 2::3]
    pose_cat =  tf.concat([x,y,z], axis = 1)
   # pose_out = tf.squeeze(pose_cat)
    return pose_cat


def weighted_pose_2d_loss(y_true, y_pred):
    # the custom loss functions weights joints separately
    # it's possible to completely ignore joint detections by setting the respective entries to zero

    diff = tf.to_float(tf.abs(y_true - y_pred))

    # weighting the joints
    weights_t = tf.to_float(
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))

    weights = tf.tile(tf.reshape(weights_t, (1, 34)), (tf.shape(y_pred)[0], 1))

    tmp = tf.multiply(weights, diff)

    loss = tf.reduce_sum(tmp, axis=1) / 34

    return loss


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def mpjpe_pred1_2(y_true, y_pred):
    x1 = y_true[:, 0::3]
    x2 = y_pred[:, 1::3]
    y1 = y_true[:, 2::3]
    y2 = y_pred[:, 0::3]
    z1 = y_true[:, 1::3]
    z2 = y_pred[:, 2::3]
    return tf.reduce_mean(tf.sqrt(tf.square(x1 - x2)+tf.square(y1-y2)+tf.square(z1 - z2)))

def mpjpe_np(y_true, y_pred):
    '''
    x1 = y_true[:, 0::3]
    x2 = y_pred[:, 0::3]
    y1 = y_true[:, 1::3]
    y2 = y_pred[:, 1::3]
    z1 = y_true[:, 2::3]
    z2 = y_pred[:, 2::3]
    return np.mean(np.sqrt(np.square(x1-x2)+np.square(y1-y2)+np.square(z1-z2)))
    '''
    return np.mean(np.sqrt(np.square(y_true - y_pred)))

def geometrical_constraint_loss(y_pred):
    x = y_pred[:, 0::3]
    y = y_pred[:, 1::3]
    z = y_pred[:, 2::3]
    # define the ratio of the arm and leg
    weights_ratio = [0.4, 0.6]

    # left limb 7->10->11->12, right limb 7->13->14->15
    left_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 12] - x[:, 13])+tf.square(y[:, 12] - y[:, 13])+tf.square(z[:, 12] - z[:, 13]))+tf.sqrt(
        tf.square(x[:, 11] - x[:, 12])+tf.square(y[:, 11] - y[:, 12])+tf.square(z[:, 11] - z[:, 12])))
    right_arm = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 14] - x[:, 15])+tf.square(y[:, 14] - y[:, 15])+tf.square(z[:, 14] - z[:, 15]))+tf.sqrt(
        tf.square(x[:, 15] - x[:, 16])+tf.square(y[:, 15] - y[:, 16])+tf.square(z[:, 15] - z[:, 16])))
    left_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 4] - x[:, 5]) + tf.square(y[:, 4] - y[:, 5]) + tf.square(z[:, 4] - z[:, 5])) + tf.sqrt(
        tf.square(x[:, 5] - x[:, 6]) + tf.square(y[:, 5] - y[:, 6]) + tf.square(z[:, 5] - z[:, 6])))
    right_leg = tf.reduce_mean(tf.sqrt(
        tf.square(x[:, 1] - x[:, 2]) + tf.square(y[:, 1] - y[:, 2]) + tf.square(z[:, 1] - z[:, 2])) + tf.sqrt(
        tf.square(x[:, 2] - x[:, 3]) + tf.square(y[:, 2] - y[:, 3]) + tf.square(z[:, 2] - z[:, 3])))
    loss = tf.reduce_mean(tf.abs(left_arm - right_arm)) * weights_ratio[0] + tf.reduce_mean(tf.abs(left_leg - right_leg)) * weights_ratio[1]
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
poses_2d, poses_3d,pose_2d_eval, pose_3d_eval = prepare_training_data()
num_joints = 17
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
# heuristic loss for 3d pose estimation
loss_3d_heuristic = Lambda(calculate_geometrical_loss)(pose_3d_1)
random_transfer_3d = Lambda(transformation_forward)(pose_3d_1)
#pose_2d_1 = reprojection_model(random_transfer_3d)
pose_2d_1 = Lambda(random_2d_projection)(random_transfer_3d)
discriminator_for_generator = discriminator(pose_2d_1)
pose_3d_2 = lifting_model(pose_2d_1)
inverse_transfer_3d = Lambda(transformation_inverse)(pose_3d_2)
loss_3d = Lambda(mpjpe_pred1_2, arguments={'y_pred': random_transfer_3d})(inverse_transfer_3d)
# Geometrical symmetry regularization for 3d pose 2
loss_3d_geometric_2 = Lambda(geometrical_constraint_loss)(pose_3d_2)
pose_2d_2 = recovered_2d_projection(inverse_transfer_3d)
#pose_2d_2 = reprojection_model(inverse_transfer_3d)
loss_2d = Lambda(weighted_pose_2d_loss, arguments={'y_pred':pose_2d_2})(pose_2d_1)


generator = Model(inputs=[pose_in_2D], outputs = [pose_2d_1])


for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

adversarial_model = Model(inputs=[pose_in_2D], outputs=[discriminator_for_generator, loss_3d_geometric_1,
                                                        loss_3d_heuristic, loss_3d_geometric_2, loss_3d, loss_2d])
# 这里的loss weights可以适当调整
adversarial_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss, diy_loss,
                                                                              diy_loss,diy_loss,diy_loss, diy_loss],
                          loss_weights=[1,0.5,1,0.5,1,1])

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


for epoch in range(70):
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
            adversarial_loss.append(adversarial_model.train_on_batch(noise, [np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1)),np.zeros((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1)),np.zeros((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))]))
    #  需要改造一下

        if i % 100 == 0 and i > 0:
            pred = lifting_model.predict(pose_2d_eval)
            # caculate training
            val = 0
            for p in range(200):
              #  val = val + 1000*err_3dpe(pose_3d_eval[p:p+1, :], pred[p:p+1, :])
                val = val + mpjpe_np(pose_3d_eval[p:p+1, :], pred[p:p+1,:])

            val = val/200
                #  loss_3d_geometric_1, loss_3d_geometric_2, loss_3d, loss_2d
            sys.stdout.write("\rIteration %d: 3d_error: %.3e, wasserstain_loss: %.3e, 3d_geometric_1,2: %.3e, %.3e, heuristic_loss: %.3e, double_lifting_error: %.3e,loss_2d:%.3e,disc_loss: %.3e\n"
                              %  (i, val, adversarial_loss[-1][0],adversarial_loss[-1][1],adversarial_loss[-1][3],adversarial_loss[-1][2], adversarial_loss[-1][4],adversarial_loss[-1][5], discriminator_loss[-1][0]))

            try:
                with open("logs/log_" + net_name + ".txt", "a") as logfile:
                    logfile.write("\rIteration %d: 3d_error: %.3e, wasserstain_loss: %.3e, 3d_geometric_1,2: %.3e, %.3e, heuristic_loss: %.3e, double_lifting_error: %.3e,loss_2d:%.3e,disc_loss: %.3e\n"
                              %  (i, val, adversarial_loss[-1][0],adversarial_loss[-1][1],adversarial_loss[-1][3],adversarial_loss[-1][2], adversarial_loss[-1][4],adversarial_loss[-1][5], discriminator_loss[-1][0]))
            except:
                print('error while writing logfile')
            sys.stdout.flush()
        # save model every 1000 iterations
        if i % 1000 ==0 and i > 0:
            lifting_model.save('models/tmp/lifting_' + net_name+'.h5')
            generator.save('models/tmp/generator_'  + net_name+'.h5')
            discriminator.save('models/tmp/discriminator_' + net_name+'.h5')
            adversarial_model.save('models/tmp/adversarial_model_' + net_name+'.h5')
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