import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
import pdb

global_dict = {}
global_dict['rotation_class_num'] = 0


def batch_distance_matrix(A):
    r = tf.reduce_sum(A * A, axis=2, keep_dims=True)
    m = tf.matmul(A, tf.transpose(A, perm=(0, 2, 1)))
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1))
    return D


# a b c
# d e f
# g h i
# a(ei − fh) − b(di − fg) + c(dh − eg)
def compute_determinant(A):
    return A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]) \
           - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0]) \
           + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])


# A shape is (N, P, 3, 3)
# return shape is (N, P, 3)
def compute_eigenvals(A):
    A_11 = A[:, :, 0, 0]  # (N, P)
    A_12 = A[:, :, 0, 1]
    A_13 = A[:, :, 0, 2]
    A_22 = A[:, :, 1, 1]
    A_23 = A[:, :, 1, 2]
    A_33 = A[:, :, 2, 2]
    I = tf.eye(3)
    p1 = tf.square(A_12) + tf.square(A_13) + tf.square(A_23)  # (N, P)
    q = tf.trace(A) / 3  # (N, P)
    p2 = tf.square(A_11 - q) + tf.square(A_22 - q) + tf.square(A_33 - q) + 2 * p1  # (N, P)
    p = tf.sqrt(p2 / 6) + 1e-8  # (N, P)
    N = tf.shape(A)[0]
    q_4d = tf.reshape(q, (N, -1, 1, 1))  # (N, P, 1, 1)
    p_4d = tf.reshape(p, (N, -1, 1, 1))
    B = (1 / p_4d) * (A - q_4d * I)  # (N, P, 3, 3)
    r = tf.clip_by_value(compute_determinant(B) / 2, -1, 1)  # (N, P)
    phi = tf.acos(r) / 3  # (N, P)
    eig1 = q + 2 * p * tf.cos(phi)  # (N, P)
    eig3 = q + 2 * p * tf.cos(phi + (2 * math.pi / 3))
    eig2 = 3 * q - eig1 - eig3
    return tf.abs(tf.stack([eig1, eig2, eig3], axis=2))  # (N, P, 3)


def compute_curvature(nn_pts):
    nn_pts_mean = tf.reduce_mean(nn_pts, axis=2, keep_dims=True)  # (N, P, 1, 3)
    nn_pts_demean = nn_pts - nn_pts_mean  # (N, P, K, 3)
    nn_pts_NPK31 = tf.expand_dims(nn_pts_demean, axis=-1)
    covariance_matrix = tf.matmul(nn_pts_NPK31, nn_pts_NPK31, transpose_b=True)  # (N, P, K, 3, 3)
    covariance_matrix_mean = tf.reduce_mean(covariance_matrix, axis=2)  # (N, P, 3, 3)
    eigvals = compute_eigenvals(covariance_matrix_mean)  # (N, P, 3)
    curvature = tf.reduce_min(eigvals, axis=-1) / (tf.reduce_sum(eigvals, axis=-1) + 1e-8)
    return curvature


def random_choice_2d(size, prob_matrix):
    n_row = prob_matrix.shape[0]
    n_col = prob_matrix.shape[1]
    choices = np.ones((n_row, size), dtype=np.int32)
    for idx_row in range(n_row):
        choices[idx_row] = np.random.choice(n_col, size=size, replace=False, p=prob_matrix[idx_row])
    return choices


def curvature_based_sample(points, k, sample_num):
    batch_size = tf.shape(points)[0]
    points_num = tf.shape(points)[1]
    D = batch_distance_matrix(points)
    distances, top_k_indices = tf.nn.top_k(-D, k=k, sorted=False)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, points_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(top_k_indices, axis=-1)], axis=-1)
    nn_pts = tf.gather_nd(points, indices)  # B,N,K,3
    # print(nn_pts.get_shape())

    curvature = compute_curvature(nn_pts)  # B,N
    # print(curvature.get_shape())

    ## top k sample
    # _, point_indices = tf.nn.top_k(curvature, k=sample_num, sorted=False)
    #
    # pts_shape = tf.shape(nn_pts)
    # batch_size = pts_shape[0]
    # batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, sample_num, 1))
    # indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=2)], axis=2)

    # prob sample
    prob_matrix = curvature / tf.reduce_sum(curvature, axis=-1, keep_dims=True)
    point_indices = tf.py_func(random_choice_2d, [sample_num, prob_matrix], tf.int32)
    point_indices.set_shape([points.get_shape()[0], sample_num])

    batch_size = tf.shape(points)[0]
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, sample_num, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=2)], axis=2)
    return indices


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    rotation_class_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, rotation_class_pl


def get_model(point_cloud, rotation_class_num, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    global_dict['rotation_class_num'] = rotation_class_num
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('rotation_dgcnn') as sc:
        net1, dgcnn_end_points = get_dgcnn_model(point_cloud, is_training, bn_decay)

    # end_points['stn_features'] = dgcnn_end_points['features']
    # print(end_points['stn_features'].get_shape())
    # pdb.set_trace()

    indices = curvature_based_sample(point_cloud, 8, num_point // 5)
    point_cloud2 = tf.gather_nd(point_cloud, indices)
    # print(point_cloud2.get_shape())
    # end_points['sampled_points'] = point_cloud2

    with tf.variable_scope('rotation_pointnet') as sc:
        net2, _ = get_dgcnn_model(point_cloud2, is_training, bn_decay)

    # print(net1.get_shape())
    # print(net2.get_shape())

    net = tf.concat([net1, net2], axis=-1)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
    #                        scope='dp1')
    net_features = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                           scope='fc2', bn_decay=bn_decay)
    end_points['stn_features'] = net_features
    # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
    #                       scope='dp2')
    net = tf_util.fully_connected(net_features, global_dict['rotation_class_num'], activation_fn=None, scope='fc3')

    return net, end_points


def get_dgcnn_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20

    point_cloud_transformed = point_cloud

    # EdgeConv1
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    # EdgeConv2
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    # EdgeConv3
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    # EdgeConv4
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    net5 = net
    end_points['features'] = net5

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)

    return net, end_points


def get_pointnet_model(point_cloud, is_training, bn_decay=None, K=3, is_dist=False):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
      Return:
        Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # adj_matrix = tf_util.pairwise_distance(point_cloud)
    # nn_idx = tf_util.knn(adj_matrix, k=20)
    # edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=20)

    edge_feature = point_cloud
    edge_feature = tf.expand_dims(edge_feature, 2)

    net = tf_util.conv2d(edge_feature, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay, is_dist=is_dist)
    net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay, is_dist=is_dist)

    # net = tf.reduce_max(net, axis=-2, keep_dims=True)

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay, is_dist=is_dist)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay, is_dist=is_dist)

    return net, end_points


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    labels = tf.one_hot(indices=label, depth=global_dict['rotation_class_num'])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


if __name__ == '__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed >= 0.5] = 1
    label_feed[label_feed < 0.5] = 0
    label_feed = label_feed.astype(np.int32)

    # # np.save('./debug/input_feed.npy', input_feed)
    # input_feed = np.load('./debug/input_feed.npy')
    # print input_feed

    with tf.Graph().as_default():
        input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
        pos, ftr = get_model(input_pl, tf.constant(True))
        # loss = get_loss(logits, label_pl, None)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {input_pl: input_feed, label_pl: label_feed}
            res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
            print(res1.shape)
            print(res1)

            print(res2.shape)
            print(res2)












