import os
import sys
import numpy as np
import h5py
import math
import pdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
# if not os.path.exists(DATA_DIR):
#   os.mkdir(DATA_DIR)
# if not os.path.exists('/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048'):
#   www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#   zipfile = os.path.basename(www)
#   os.system('wget %s; unzip %s' % (www, zipfile))
#   os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#   os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,... numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


# edited #################################

def rotate_point_cloud_by_so3(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  batch_size = batch_data.shape[0]
  rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
  rotation_vectors = np.zeros((batch_size, 3), dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle_x = np.random.uniform() * 2 * np.pi
    rotation_angle_y = np.random.uniform() * 2 * np.pi
    rotation_angle_z = np.random.uniform() * 2 * np.pi
    rotation_vectors[k, 0] = rotation_angle_x
    rotation_vectors[k, 1] = rotation_angle_y
    rotation_vectors[k, 2] = rotation_angle_z
    cosval_x = np.cos(rotation_angle_x)
    sinval_x = np.sin(rotation_angle_x)
    cosval_y = np.cos(rotation_angle_y)
    sinval_y = np.sin(rotation_angle_y)
    cosval_z = np.cos(rotation_angle_z)
    sinval_z = np.sin(rotation_angle_z)
    rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                  [0, 1, 0],
                                  [-sinval_y, 0, cosval_y]])
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cosval_x, -sinval_x],
                                  [0, sinval_x, cosval_x]])
    rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                  [sinval_z, cosval_z, 0],
                                  [0, 0, 1]])
    shape_pc = batch_data[k, ...]
    rotation_matrix = np.dot(rotation_matrix_y.transpose(), rotation_matrix_x.transpose())
    rotation_matrix = np.dot(rotation_matrix, rotation_matrix_z.transpose())
    rotation_matrics[k, :, :] = rotation_matrix.copy()
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data, rotation_matrics, rotation_vectors


def rotate_point_cloud_by_so3_2(batch_data):
  batch_size = batch_data.shape[0]
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
  rotation_vectors = np.zeros((batch_size, 3), dtype=np.float32)
  for k in range(batch_data.shape[0]):
    x1, x2, x3 = np.random.rand(3)
    rotation_vectors[k, 0] = x1
    rotation_vectors[k, 1] = x2
    rotation_vectors[k, 2] = x3
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), M)
    rotation_matrics[k, :, :] = M.copy()
  return rotated_data, rotation_matrics, rotation_vectors


def rotate_point_cloud_by_so3_3(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  batch_size = batch_data.shape[0]
  rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
  rotation_vectors = np.zeros((batch_size, 3), dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle_y1 = np.random.uniform() * 2 * np.pi
    rotation_angle_x = np.random.uniform() * np.pi
    rotation_angle_y2 = np.random.uniform() * 2 * np.pi
    rotation_vectors[k, 0] = rotation_angle_y1
    rotation_vectors[k, 1] = rotation_angle_x
    rotation_vectors[k, 2] = rotation_angle_y2
    cosval_y1 = np.cos(rotation_angle_y1)
    sinval_y1 = np.sin(rotation_angle_y1)
    cosval_x = np.cos(rotation_angle_x)
    sinval_x = np.sin(rotation_angle_x)
    cosval_y2 = np.cos(rotation_angle_y2)
    sinval_y2 = np.sin(rotation_angle_y2)
    rotation_matrix_y1 = np.array([[cosval_y1, 0, sinval_y1],
                                  [0, 1, 0],
                                  [-sinval_y1, 0, cosval_y1]])
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cosval_x, -sinval_x],
                                  [0, sinval_x, cosval_x]])
    rotation_matrix_y2 = np.array([[cosval_y2, 0, sinval_y2],
                                  [0, 1, 0],
                                  [-sinval_y2, 0, cosval_y2]])
    shape_pc = batch_data[k, ...]
    rotation_matrix = np.dot(rotation_matrix_y1.transpose(), rotation_matrix_x.transpose())
    rotation_matrix = np.dot(rotation_matrix, rotation_matrix_y2.transpose())
    rotation_matrics[k, :, :] = rotation_matrix.copy()
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data, rotation_matrics, rotation_vectors


def rotate_point_cloud_by_so3_small_angle(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
  """
  small_angle = np.pi / 4
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  batch_size = batch_data.shape[0]
  rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
  rotation_vectors = np.zeros((batch_size, 3), dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle_x = np.random.uniform() * small_angle
    rotation_angle_y = np.random.uniform() * small_angle
    rotation_angle_z = np.random.uniform() * small_angle
    rotation_angle_x = rotation_angle_x - small_angle / 2
    rotation_angle_y = rotation_angle_y - small_angle / 2
    rotation_angle_z = rotation_angle_z - small_angle / 2
    rotation_vectors[k, 0] = rotation_angle_x
    rotation_vectors[k, 1] = rotation_angle_y
    rotation_vectors[k, 2] = rotation_angle_z
    cosval_x = np.cos(rotation_angle_x)
    sinval_x = np.sin(rotation_angle_x)
    cosval_y = np.cos(rotation_angle_y)
    sinval_y = np.sin(rotation_angle_y)
    cosval_z = np.cos(rotation_angle_z)
    sinval_z = np.sin(rotation_angle_z)
    rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                  [0, 1, 0],
                                  [-sinval_y, 0, cosval_y]])
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cosval_x, -sinval_x],
                                  [0, sinval_x, cosval_x]])
    rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                  [sinval_z, cosval_z, 0],
                                  [0, 0, 1]])
    shape_pc = batch_data[k, ...]
    rotation_matrix = np.dot(rotation_matrix_y.transpose(), rotation_matrix_x.transpose())
    rotation_matrix = np.dot(rotation_matrix, rotation_matrix_z.transpose())
    rotation_matrics[k, :, :] = rotation_matrix.copy()
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data, rotation_matrics, rotation_vectors


def rotate_point_cloud_with_normal_by_so3(batch_data):
  batch_size = batch_data.shape[0]
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
  rotation_vectors = np.zeros((batch_size, 3), dtype=np.float32)
  for k in range(batch_data.shape[0]):
    x1, x2, x3 = np.random.rand(3)
    rotation_vectors[k, 0] = x1
    rotation_vectors[k, 1] = x2
    rotation_vectors[k, 2] = x3
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    # shape_pc = batch_data[k, ...]
    rotated_data[k, :, :3] = np.dot(batch_data[k, :, :3].reshape((-1, 3)), M)
    rotated_data[k, :, 3:] = np.dot(batch_data[k, :, 3:].reshape((-1, 3)), M)
    rotation_matrics[k, :, :] = M.copy()
  return rotated_data, rotation_matrics, rotation_vectors

##########################################


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data


def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)


def loadDataFile(filename):
  return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:] # (2048, 2048, 3)
  label = f['label'][:] # (2048, 1)
  seg = f['pid'][:] # (2048, 2048)
  return (data, label, seg)


# edited #################################

def transform_to_spherical_coordinates(points):
    r = np.sqrt(np.sum(points*points, axis=-1))
    theta = np.arccos(points[:,:,2]/r)
    varphi = np.arctan2(points[:,:,1], points[:,:,0])
    return np.stack((r, theta, varphi), axis=-1)


def r_theta_to_rotation_matrix(r, theta):
  transform_list = []
  # transform_list.append(r[:, 0] * r[:, 0] * (1 - np.cos(theta)) + np.cos(theta))
  # transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) + r[:, 2] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) - r[:, 1] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) - r[:, 2] * np.sin(theta))
  # transform_list.append(r[:, 1] * r[:, 1] * (1 - np.cos(theta)) + np.cos(theta))
  # transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) + r[:, 0] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) + r[:, 1] * np.sin(theta))
  # transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) - r[:, 0] * np.sin(theta))
  # transform_list.append(r[:, 2] * r[:, 2] * (1 - np.cos(theta)) + np.cos(theta))

  transform_list.append(r[0] * r[0] * (1 - np.cos(theta)) + np.cos(theta))
  transform_list.append(r[0] * r[1] * (1 - np.cos(theta)) - r[2] * np.sin(theta))
  transform_list.append(r[0] * r[2] * (1 - np.cos(theta)) + r[1] * np.sin(theta))
  transform_list.append(r[0] * r[1] * (1 - np.cos(theta)) + r[2] * np.sin(theta))
  transform_list.append(r[1] * r[1] * (1 - np.cos(theta)) + np.cos(theta))
  transform_list.append(r[1] * r[2] * (1 - np.cos(theta)) - r[0] * np.sin(theta))
  transform_list.append(r[0] * r[2] * (1 - np.cos(theta)) - r[1] * np.sin(theta))
  transform_list.append(r[1] * r[2] * (1 - np.cos(theta)) + r[0] * np.sin(theta))
  transform_list.append(r[2] * r[2] * (1 - np.cos(theta)) + np.cos(theta))
  rotation_matrix = np.stack(transform_list, axis=0)
  rotation_matrix = np.reshape(rotation_matrix, [3, 3])
  return rotation_matrix


def rotation_matrix_to_r_theta(rotation_matrix):
  theta = (rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2] - 1) / 2
  if theta > 1:
      theta = 1
  if theta < -1:
      theta = -1
  theta = np.arccos(theta)
  # print(theta)
  r = 1 / (2 * np.sin(theta) + 1e-3) * np.stack(
    (rotation_matrix[2, 1] - rotation_matrix[1, 2],
     rotation_matrix[0, 2] - rotation_matrix[2, 0],
     rotation_matrix[1, 0] - rotation_matrix[0, 1]), axis=0)
  # print(r)
  rotation_vector = r * theta
  return rotation_vector


def r_theta_to_rotation_matrics(rotation_vectors):
  batch_size = rotation_vectors.shape[0]
  theta = np.sqrt(np.sum(rotation_vectors * rotation_vectors, axis=-1)) + 1e-3
  r = rotation_vectors / np.expand_dims(theta, axis=-1)
  transform_list = []
  # transform_list.append(r[:, 0] * r[:, 0] * (1 - np.cos(theta)) + np.cos(theta))
  # transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) + r[:, 2] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) - r[:, 1] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) - r[:, 2] * np.sin(theta))
  # transform_list.append(r[:, 1] * r[:, 1] * (1 - np.cos(theta)) + np.cos(theta))
  # transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) + r[:, 0] * np.sin(theta))
  # transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) + r[:, 1] * np.sin(theta))
  # transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) - r[:, 0] * np.sin(theta))
  # transform_list.append(r[:, 2] * r[:, 2] * (1 - np.cos(theta)) + np.cos(theta))

  transform_list.append(r[:, 0] * r[:, 0] * (1 - np.cos(theta)) + np.cos(theta))
  transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) - r[:, 2] * np.sin(theta))
  transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) + r[:, 1] * np.sin(theta))
  transform_list.append(r[:, 0] * r[:, 1] * (1 - np.cos(theta)) + r[:, 2] * np.sin(theta))
  transform_list.append(r[:, 1] * r[:, 1] * (1 - np.cos(theta)) + np.cos(theta))
  transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) - r[:, 0] * np.sin(theta))
  transform_list.append(r[:, 0] * r[:, 2] * (1 - np.cos(theta)) - r[:, 1] * np.sin(theta))
  transform_list.append(r[:, 1] * r[:, 2] * (1 - np.cos(theta)) + r[:, 0] * np.sin(theta))
  transform_list.append(r[:, 2] * r[:, 2] * (1 - np.cos(theta)) + np.cos(theta))
  rotation_matrics = np.stack(transform_list, axis=1)
  rotation_matrics = np.reshape(rotation_matrics, [batch_size, 3, 3])
  return rotation_matrics


# R = np.array([[[-1,0,0],[0,-1,0],[0,0,1]]], dtype=np.float32)
def rotation_matrics_to_r_theta(rotation_matrics):
  theta = (rotation_matrics[:, 0, 0] + rotation_matrics[:, 1, 1] + rotation_matrics[:, 2, 2] - 1) / 2
  theta[theta > 1] = 1
  theta[theta < -1] = -1
  theta = np.arccos(theta)
  # print(theta)
  r = 1 / (2 * np.expand_dims(np.sin(theta), axis=-1) + 1e-6) * np.stack(
    (rotation_matrics[:, 2, 1] - rotation_matrics[:, 1, 2],
     rotation_matrics[:, 0, 2] - rotation_matrics[:, 2, 0],
     rotation_matrics[:, 1, 0] - rotation_matrics[:, 0, 1]), axis=1)
  # print(r)
  rotation_vectors = r * np.expand_dims(theta, axis=-1)
  return rotation_vectors
# print(rotation_matrix_to_r_theta(R))
# pdb.set_trace()


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rotation_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
        dtype=q.dtype)
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrix):
    K = np.zeros((4,4), dtype=np.float32)
    R = rotation_matrix
    K[1-1,1-1] = (1/3) * (R[1-1,1-1] - R[2-1,2-1] - R[3-1,3-1])
    K[1-1,2-1] = (1/3) * (R[2-1,1-1] + R[1-1,2-1])
    K[1-1,3-1] = (1/3) * (R[3-1,1-1] + R[1-1,3-1])
    K[1-1,4-1] = (1/3) * (R[2-1,3-1] - R[3-1,2-1])
    K[2-1,1-1] = (1/3) * (R[2-1,1-1] + R[1-1,2-1])
    K[2-1,2-1] = (1/3) * (R[2-1,2-1] - R[1-1,1-1] - R[3-1,3-1])
    K[2-1,3-1] = (1/3) * (R[3-1,2-1] + R[2-1,3-1])
    K[2-1,4-1] = (1/3) * (R[3-1,1-1] - R[1-1,3-1])
    K[3-1,1-1] = (1/3) * (R[3-1,1-1] + R[1-1,3-1])
    K[3-1,2-1] = (1/3) * (R[3-1,2-1] + R[2-1,3-1])
    K[3-1,3-1] = (1/3) * (R[3-1,3-1] - R[1-1,1-1] - R[2-1,2-1])
    K[3-1,4-1] = (1/3) * (R[1-1,2-1] - R[2-1,1-1])
    K[4-1,1-1] = (1/3) * (R[2-1,3-1] - R[3-1,2-1])
    K[4-1,2-1] = (1/3) * (R[3-1,1-1] - R[1-1,3-1])
    K[4-1,3-1] = (1/3) * (R[1-1,2-1] - R[2-1,1-1])
    K[4-1,4-1] = (1/3) * (R[1-1,1-1] + R[2-1,2-1] + R[3-1,3-1])
    w, v = np.linalg.eig(K)
    index = np.argmax(w)
    # print(w)
    # print(v)
    # [V,D] = eig(K)
    q = v[:, index]
    q = np.array([q[4-1], q[1-1], q[2-1], q[3-1]], dtype=np.float32)
    return q


# qs = np.array([[0.70710677, 0., 0., 0.70710677]], dtype=np.float32)
def quaternions_to_rotation_matrics(quats):
    batch_size = quats.shape[0]
    rotation_matrics = np.zeros((batch_size, 3, 3), dtype=np.float32)
    for i in range(batch_size):
        q = quats[i].copy()
        n = np.dot(q, q)
        if n < np.finfo(q.dtype).eps:
            return np.identity(4)
        q = q * np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrics[i, :, :] = np.array(
            [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
             [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
             [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
            dtype=q.dtype)
    return rotation_matrics
# print(quaternion_to_rotation_matrix(qs))
# pdb.set_trace()


# R = np.array([[[0,1,0],[-1,0,0],[0,0,1]]], dtype=np.float32)
def rotation_matrics_to_quaternions(rotation_matrics):
    batch_size = rotation_matrics.shape[0]
    q = np.zeros((batch_size, 4), dtype=np.float32)
    K = np.zeros((4,4), dtype=np.float32)
    for i in range(batch_size):
        R = rotation_matrics[i]
        K[1-1,1-1] = (1/3) * (R[1-1,1-1] - R[2-1,2-1] - R[3-1,3-1])
        K[1-1,2-1] = (1/3) * (R[2-1,1-1] + R[1-1,2-1])
        K[1-1,3-1] = (1/3) * (R[3-1,1-1] + R[1-1,3-1])
        K[1-1,4-1] = (1/3) * (R[2-1,3-1] - R[3-1,2-1])
        K[2-1,1-1] = (1/3) * (R[2-1,1-1] + R[1-1,2-1])
        K[2-1,2-1] = (1/3) * (R[2-1,2-1] - R[1-1,1-1] - R[3-1,3-1])
        K[2-1,3-1] = (1/3) * (R[3-1,2-1] + R[2-1,3-1])
        K[2-1,4-1] = (1/3) * (R[3-1,1-1] - R[1-1,3-1])
        K[3-1,1-1] = (1/3) * (R[3-1,1-1] + R[1-1,3-1])
        K[3-1,2-1] = (1/3) * (R[3-1,2-1] + R[2-1,3-1])
        K[3-1,3-1] = (1/3) * (R[3-1,3-1] - R[1-1,1-1] - R[2-1,2-1])
        K[3-1,4-1] = (1/3) * (R[1-1,2-1] - R[2-1,1-1])
        K[4-1,1-1] = (1/3) * (R[2-1,3-1] - R[3-1,2-1])
        K[4-1,2-1] = (1/3) * (R[3-1,1-1] - R[1-1,3-1])
        K[4-1,3-1] = (1/3) * (R[1-1,2-1] - R[2-1,1-1])
        K[4-1,4-1] = (1/3) * (R[1-1,1-1] + R[2-1,2-1] + R[3-1,3-1])
        w, v = np.linalg.eig(K)
        index = np.argmax(w)
        # print(w)
        # print(v)
        # [V,D] = eig(K)
        q[i, :] = v[:, index]
        q[i, :] = np.array([q[i, 4-1], q[i, 1-1], q[i, 2-1], q[i, 3-1]], dtype=np.float32)
    return q
# print(rotation_matrix_to_quaternion(R))
# pdb.set_trace()


def get_rotation_matrix(rotation_vector):
  cosval_x = np.cos(rotation_vector[0])
  sinval_x = np.sin(rotation_vector[0])
  cosval_y = np.cos(rotation_vector[1])
  sinval_y = np.sin(rotation_vector[1])
  cosval_z = np.cos(rotation_vector[2])
  sinval_z = np.sin(rotation_vector[2])
  rotation_matrix_x = np.array([[1, 0, 0],
                                [0, cosval_x, -sinval_x],
                                [0, sinval_x, cosval_x]])
  rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                [0, 1, 0],
                                [-sinval_y, 0, cosval_y]])
  rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                [sinval_z, cosval_z, 0],
                                [0, 0, 1]])
  rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_y.transpose())
  rotation_matrix = np.dot(rotation_matrix, rotation_matrix_z.transpose())
  return rotation_matrix


def get_rotation_matrics(rotation_vectors):
  batch_size = rotation_vectors.shape[0]
  rotation_matrices = np.zeros((batch_size, 3, 3), dtype=np.float32)
  for i in range(rotation_vectors.shape[0]):
    rotation_matrices[i, :, :] = get_rotation_matrix(rotation_vectors[i])
  return rotation_matrices


# z rotation classification
def get_z_rotation_matrics_all():
    angle_cell = np.pi / 6
    rotation_vectors_all = np.zeros((0, 3), dtype=np.float32)
    rotation_vectors_all = np.vstack((rotation_vectors_all, np.array([0, 0, 0], dtype=np.float32)))
    for i in range(1, int(np.pi // angle_cell)):
        for j in range(2 * int(np.pi // angle_cell)):
            rotation_vector = np.array([i * angle_cell, j * angle_cell, 0], dtype=np.float32)
            rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
    rotation_vectors_all = np.vstack((rotation_vectors_all, np.array([np.pi, 0, 0], dtype=np.float32)))
    rotation_class_num = 2 + int(np.pi // angle_cell - 1) * int(2 * np.pi // angle_cell)  # 62

    # class_num = 14
    # rotation_vectors_all = \
    #     [[0, 0, 0],
    #      [math.pi / 2, 0, 0],
    #      [math.pi, 0, 0],
    #      [3 * math.pi / 2, 0, 0],
    #      [0, 0, math.pi / 2],
    #      [0, 0, 3 * math.pi / 2],
    #      [math.pi / 3, math.pi / 4, 0],
    #      [math.pi / 3, 3 * math.pi / 4, 0],
    #      [math.pi / 3, 5 * math.pi / 4, 0],
    #      [math.pi / 3, 7 * math.pi / 4, 0],
    #      [2 * math.pi / 3, math.pi / 4, 0],
    #      [2 * math.pi / 3, 3 * math.pi / 4, 0],
    #      [2 * math.pi / 3, 5 * math.pi / 4, 0],
    #      [2 * math.pi / 3, 7 * math.pi / 4, 0]
    #      ]

    rotation_vectors_all = np.array(rotation_vectors_all, np.float32)
    rotation_matrics_all = np.zeros((rotation_class_num, 3, 3), dtype=np.float32)
    rotation_y_all = np.zeros((rotation_class_num, 3), dtype=np.float32)
    for i in range(rotation_class_num):
        rotation_matrics_all[i, :, :] = get_rotation_matrix(rotation_vectors_all[i, :])
        rotation_y_all[i, :] = np.dot(np.array([0, 1, 0], np.float32), rotation_matrics_all[i, :, :])
    return rotation_class_num, rotation_matrics_all, rotation_y_all


def get_z_rotation_class(rotation_vectors, rotation_y_all):
    rotation_class = np.zeros((rotation_vectors.shape[0]), dtype=np.int32)
    for i in range(rotation_vectors.shape[0]):
        rotation_angle_x = rotation_vectors[i, 1]
        rotation_angle_y2 = rotation_vectors[i, 2]
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        cosval_y2 = np.cos(rotation_angle_y2)
        sinval_y2 = np.sin(rotation_angle_y2)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix_y2 = np.array([[cosval_y2, 0, sinval_y2],
                                       [0, 1, 0],
                                       [-sinval_y2, 0, cosval_y2]])
        rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_y2.transpose())

        temp = np.array([0, 1, 0], np.float32)
        temp2 = np.dot(temp, rotation_matrix)
        distance = np.linalg.norm(np.expand_dims(temp2, axis=0) - rotation_y_all, axis=1)
        rotation_class[i] = np.argmin(distance)

        # min_r = 1e6
        # min_index = 0
        # for j in range(rotation_matrics_all.shape[0]):
        #     # diff_r = rotation_matrix * rotation_matrics_all[j].transpose()
        #     # r = np.arccos((np.trace(diff_r) - 1) / 2) * 180 / np.pi
        #     r = np.linalg.norm(temp2 - rotation_y_all[j])
        #     if r < min_r:
        #         min_r = r
        #         min_index = j
        # rotation_class[i] = min_index

        # M_expand = np.tile(np.expand_dims(rotation_matrix, axis=0), (6, 1, 1))
        # M_distance = np.sum(np.sum(np.power(M_expand - rotation_matrics_all, 2), axis=2), axis=1)
        # rotation_class[i] = np.argmin(M_distance)

        # diff = np.reshape(rotation_matrics_all, [6, 9]) - np.expand_dims(np.reshape(rotation_matrix, [9]), axis=0)
        # distance = np.linalg.norm(diff, axis=1)
        # rotation_class[i] = np.argmin(distance)
    return rotation_class


def rotate_data_to_origin_classification_z(current_data, rotation_matrics_all, current_rotation_class):
  current_rotated_data = []
  for shape_index in range(current_data.shape[0]):
    rotation_index = current_rotation_class[shape_index]
    rotated_data = np.dot(current_data[shape_index], rotation_matrics_all[rotation_index].transpose())
    current_rotated_data.append(rotated_data)
  current_rotated_data = np.stack(current_rotated_data, axis=0)
  return current_rotated_data


def get_rotation_matrix_of_two_vectors(vector1, vector2):
  norm_vector1 = np.linalg.norm(vector1)
  norm_vector2 = np.linalg.norm(vector2)
  theta = np.arccos(np.dot(vector1, vector2.transpose()) / (norm_vector1 * norm_vector2))

  r = np.array([0, 0, 0], np.float32)
  r[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1]
  r[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2]
  r[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0]
  r = r / np.linalg.norm(r)

  #print(theta)
  # print(r)
  rotation_matrix = r_theta_to_rotation_matrix(r, theta)
  return rotation_matrix


# so3 rotation classification
def get_so3_rotation_matrics_all():
    Ry = []
    rotation_plane_all = []
    y_angle = np.pi / 6
    y_num = int(2 * np.pi / y_angle)
    for i in range(y_num):
        rotation_angle_y = i * np.pi / y_angle
        cosval_y = np.cos(rotation_angle_y)
        sinval_y = np.sin(rotation_angle_y)
        Ry.append(np.array([[cosval_y, 0, sinval_y],
                                      [0, 1, 0],
                                      [-sinval_y, 0, cosval_y]]).transpose())
        rotation_plane_all.append(np.dot(np.array([0, 0, 1], np.float32), Ry[-1]))
    rotation_plane_all = np.stack(rotation_plane_all, axis=0)

    angle_cell = np.pi / 6
    rotation_matrics_all = []
    rotation_matrics_sphere_all = []
    rotation_y_all = []
    rotation_matrix = get_rotation_matrix(np.array([0, 0, 0], dtype=np.float32))
    rotation_matrics_sphere_all.append(rotation_matrix)
    rotation_y_all.append(np.dot(np.array([0, 1, 0], np.float32), rotation_matrix))
    for k in range(y_num):
        rotation_matrics_all.append(np.dot(Ry[k], rotation_matrix))

    for i in range(1, int(np.pi // angle_cell)):
        for j in range(2 * int(np.pi // angle_cell)):
            rotation_vector = np.array([i * angle_cell, j * angle_cell, 0], dtype=np.float32)
            rotation_matrix = get_rotation_matrix(rotation_vector)
            rotation_matrics_sphere_all.append(rotation_matrix)
            rotation_y_all.append(np.dot(np.array([0, 1, 0], np.float32), rotation_matrix))
            for k in range(y_num):
                rotation_matrics_all.append(np.dot(Ry[k], rotation_matrix))

    rotation_matrix = get_rotation_matrix(np.array([np.pi, 0, 0], dtype=np.float32))
    rotation_matrics_sphere_all.append(rotation_matrix)
    rotation_y_all.append(np.dot(np.array([0, 1, 0], np.float32), rotation_matrix))
    for k in range(y_num):
        rotation_matrics_all.append(np.dot(Ry[k], rotation_matrix))

    rotation_class_num = y_num * (2 + int(np.pi // angle_cell - 1) * int(2 * np.pi // angle_cell))  # 62*4
    rotation_matrics_all = np.stack(rotation_matrics_all, axis=0)
    rotation_matrics_sphere_all = np.stack(rotation_matrics_sphere_all, axis=0)
    rotation_y_all = np.stack(rotation_y_all, axis=0)
    print(rotation_matrics_all.shape)
    print(rotation_plane_all.shape)
    print(rotation_matrics_sphere_all.shape)
    print(rotation_y_all.shape)
    print(rotation_class_num)

    return rotation_class_num, rotation_matrics_all, rotation_plane_all, rotation_matrics_sphere_all, rotation_y_all


def get_so3_rotation_class(rotation_matrics, rotation_matrics_all, rotation_plane_all, rotation_matrics_sphere_all, rotation_y_all):
    y_num = rotation_plane_all.shape[0]
    rotation_class = np.zeros((rotation_matrics.shape[0]), dtype=np.int32)
    for i in range(rotation_matrics.shape[0]):
        rotation_matrix = rotation_matrics[i]

        # 先看是球面哪个类
        temp = np.array([0, 1, 0], np.float32)
        temp2 = np.dot(temp, rotation_matrix)
        distance = np.linalg.norm(np.expand_dims(temp2, axis=0) - rotation_y_all, axis=1)
        y_rotation_class = np.argmin(distance)

        # 再看是平面上哪个类
        R_plane = np.dot(rotation_matrix, rotation_matrics_sphere_all[y_rotation_class].transpose())
        temp = np.array([0, 0, 1], np.float32)
        temp2 = np.dot(temp, R_plane)
        distance = np.linalg.norm(np.expand_dims(temp2, axis=0) - rotation_plane_all, axis=1)
        plane_rotation_class = np.argmin(distance)
        rotation_class[i] = y_rotation_class * y_num + plane_rotation_class


        # min_r = 1e6
        # min_index = 0
        # for j in range(rotation_matrics_all.shape[0]):
        #     r = np.linalg.norm(rotation_matrix - rotation_matrics_all[j])
        #     # diff_r = rotation_matrix * rotation_matrics_all[j].transpose()
        #     # r = np.arccos((np.trace(diff_r) - 1) / 2) * 180 / np.pi
        #     if r < min_r:
        #         min_r = r
        #         min_index = j
        # rotation_class[i] = min_index
        # print(rotation_class[:50])

        # M_expand = np.tile(np.expand_dims(rotation_matrix, axis=0), (rotation_matrics_all.shape[0], 1, 1))
        # M_distance = np.sum(np.sum(np.power(M_expand - rotation_matrics_all, 2), axis=2), axis=1)
        # rotation_class[i] = np.argmin(M_distance)

        # diff = np.reshape(rotation_matrics_all, [rotation_matrics_all.shape[0], 9]) - np.expand_dims(np.reshape(rotation_matrix, [9]), axis=0)
        # distance = np.linalg.norm(diff, axis=1)
        # rotation_class[i] = np.argmin(distance)
    return rotation_class


def rotate_data_to_origin_classification_so3(current_data, rotation_matrics_all, current_rotation_class):
  current_rotated_data = []
  for shape_index in range(current_data.shape[0]):
    rotation_index = current_rotation_class[shape_index]
    rotated_data = np.dot(current_data[shape_index], rotation_matrics_all[rotation_index].transpose())
    current_rotated_data.append(rotated_data)
  current_rotated_data = np.stack(current_rotated_data, axis=0)
  return current_rotated_data


def rotate_data_to_origin_classification_so3_with_normal(current_data, rotation_matrics_all, current_rotation_class):
  current_rotated_data = []
  for shape_index in range(current_data.shape[0]):
    rotation_index = current_rotation_class[shape_index]
    rotated_data = np.dot(current_data[shape_index, :, :3], rotation_matrics_all[rotation_index].transpose())
    rotated_normal = np.dot(current_data[shape_index, :, 3:], rotation_matrics_all[rotation_index].transpose())
    current_rotated_data.append(np.concatenate((rotated_data, rotated_normal), axis=-1))
  current_rotated_data = np.stack(current_rotated_data, axis=0)
  return current_rotated_data


# z rotation regression
def get_z_rotation_vector_matrix(rotation_vectors):
    rotation_vectors2 = np.zeros((rotation_vectors.shape[0], 9), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_angle_x = rotation_vectors[i, 0]
        rotation_angle_z = rotation_vectors[i, 2]
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        cosval_z = np.cos(rotation_angle_z)
        sinval_z = np.sin(rotation_angle_z)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_z.transpose())

        rotation_vectors2[i, :] = np.reshape(rotation_matrix, [-1])
    return rotation_vectors2


def get_z_rotation_vector_quaternion(rotation_vectors):
    rotation_vectors2 = np.zeros((rotation_vectors.shape[0], 4), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_angle_x = rotation_vectors[i, 0]
        rotation_angle_z = rotation_vectors[i, 2]
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        cosval_z = np.cos(rotation_angle_z)
        sinval_z = np.sin(rotation_angle_z)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_z.transpose())

        quaternion = rotation_matrix_to_quaternion(rotation_matrix)

        rotation_vectors2[i, :] = quaternion.copy()
    return rotation_vectors2


def get_z_rotation_vector_r_theta(rotation_vectors):
    rotation_vectors2 = np.zeros((rotation_vectors.shape[0], 3), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_angle_x = rotation_vectors[i, 0]
        rotation_angle_z = rotation_vectors[i, 2]
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        cosval_z = np.cos(rotation_angle_z)
        sinval_z = np.sin(rotation_angle_z)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_z.transpose())

        r_theta = rotation_matrix_to_r_theta(rotation_matrix)

        rotation_vectors2[i, :] = r_theta.copy()
    return rotation_vectors2


def get_z_rotation_vector_euler_angle(rotation_vectors):
    rotation_vectors2 = np.zeros((rotation_vectors.shape[0], 2), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_angle_x = rotation_vectors[i, 0]
        rotation_angle_z = rotation_vectors[i, 2]
        # cosval_x = np.cos(rotation_angle_x)
        # sinval_x = np.sin(rotation_angle_x)
        # cosval_z = np.cos(rotation_angle_z)
        # sinval_z = np.sin(rotation_angle_z)
        # rotation_matrix_x = np.array([[1, 0, 0],
        #                               [0, cosval_x, -sinval_x],
        #                               [0, sinval_x, cosval_x]])
        # rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
        #                               [sinval_z, cosval_z, 0],
        #                               [0, 0, 1]])
        # rotation_matrix = np.dot(rotation_matrix_x.transpose(), rotation_matrix_z.transpose())

        euler_angle = np.array([rotation_angle_x, rotation_angle_z])

        rotation_vectors2[i, :] = euler_angle.copy()
    return rotation_vectors2


def rotate_data_to_origin_regression_z_matrix(current_data, rotation_vectors2):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        rotation_matrix_to_origin = np.reshape(rotation_vectors2[shape_index], [3, 3]).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_z_quaternion(current_data, rotation_vectors2):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        quaternion = rotation_vectors2[shape_index].copy()
        quaternion = quaternion / np.linalg.norm(quaternion)
        rotation_matrix_to_origin = quaternion_to_rotation_matrix(quaternion).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_z_r_theta(current_data, rotation_vectors2):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        r_theta = rotation_vectors2[shape_index].copy()
        theta = np.linalg.norm(r_theta)
        r = r_theta / theta
        rotation_matrix_to_origin = r_theta_to_rotation_matrix(r, theta).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_z_euler_angle(current_data, rotation_vectors2):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        euler_angle = rotation_vectors2[shape_index].copy()
        rotation_angle_x = euler_angle[0]
        rotation_angle_z = euler_angle[1]
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        cosval_z = np.cos(rotation_angle_z)
        sinval_z = np.sin(rotation_angle_z)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_matrix_to_origin = np.dot(rotation_matrix_z, rotation_matrix_x)
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_z2(current_data, rotation_vectors2):
    current_rotated_data = []
    temp = np.array([0, 1, 0], np.float32)
    for shape_index in range(current_data.shape[0]):
        spherical_coordinate = rotation_vectors2[shape_index].copy()
        if spherical_coordinate[0] > np.pi:
            spherical_coordinate[0] = np.pi
        if spherical_coordinate[0] < 0:
            spherical_coordinate[0] = 0
        if spherical_coordinate[1] < -np.pi:
            spherical_coordinate[1] = -np.pi
        if spherical_coordinate[1] > np.pi:
            spherical_coordinate[1] = np.pi
        rotated_y = np.array([0, 0, 0], np.float32)
        rotated_y[0] = np.sin(spherical_coordinate[0]) * np.cos(spherical_coordinate[1])
        rotated_y[1] = np.sin(spherical_coordinate[0]) * np.sin(spherical_coordinate[1])
        rotated_y[2] = np.cos(spherical_coordinate[0])
        rotated_y = rotated_y / np.linalg.norm(rotated_y)
        rotation_matrix_to_origin = get_rotation_matrix_of_two_vectors(rotated_y, temp).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def get_so3_rotation_vector_matrix(rotation_matrics):
    rotation_vectors = np.zeros((rotation_matrics.shape[0], 9), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_matrix = rotation_matrics[i]
        rotation_vectors[i, :] = np.reshape(rotation_matrix, [-1])
    return rotation_vectors


def get_so3_rotation_vector_quaternion(rotation_matrics):
    rotation_vectors = np.zeros((rotation_matrics.shape[0], 4), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_matrix = rotation_matrics[i]
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        rotation_vectors[i, :] = quaternion.copy()
    return rotation_vectors


def get_so3_rotation_vector_r_theta(rotation_matrics):
    rotation_vectors = np.zeros((rotation_matrics.shape[0], 3), dtype=np.float32)
    for i in range(rotation_vectors.shape[0]):
        rotation_matrix = rotation_matrics[i]
        r_theta = rotation_matrix_to_r_theta(rotation_matrix)
        rotation_vectors[i, :] = r_theta.copy()
    return rotation_vectors


def rotate_data_to_origin_regression_so3_matrix(current_data, rotation_vectors):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        rotation_matrix_to_origin = np.reshape(rotation_vectors[shape_index], [3, 3]).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_so3_quaternion(current_data, rotation_vectors):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        quaternion = rotation_vectors[shape_index].copy()
        quaternion = quaternion / np.linalg.norm(quaternion)
        rotation_matrix_to_origin = quaternion_to_rotation_matrix(quaternion).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


def rotate_data_to_origin_regression_so3_r_theta(current_data, rotation_vectors):
    current_rotated_data = []
    for shape_index in range(current_data.shape[0]):
        r_theta = rotation_vectors[shape_index].copy()
        theta = np.linalg.norm(r_theta) + 1e-3
        r = r_theta / theta
        rotation_matrix_to_origin = r_theta_to_rotation_matrix(r, theta).transpose()
        rotated_data = np.dot(current_data[shape_index], rotation_matrix_to_origin)
        current_rotated_data.append(rotated_data)
    current_rotated_data = np.stack(current_rotated_data, axis=0)
    return current_rotated_data


# data
def loadDataFile_list_class(filename_list, class_indices):
  file_data = []
  file_label = []
  for filename in filename_list:
    class_data = []
    class_label = []
    current_data, current_label = load_h5(filename)
    label = 0
    for class_index in class_indices:
      indices = (current_label == class_index)
      current_class_data = current_data[np.squeeze(indices), :, :]
      # current_class_label = current_label[np.squeeze(indices), :]
      current_class_label = np.ones((current_class_data.shape[0], 1), np.int32) * label
      label = label + 1
      class_data.append(current_class_data)
      class_label.append(current_class_label)
    file_data.append(np.concatenate(class_data, axis=0))
    file_label.append(np.concatenate(class_label, axis=0))

  return file_data, file_label


def loadDataFile_list_all(filename_list):
  data = []
  label = []
  for filename in filename_list:
    current_data, current_label = load_h5(filename)
    # print(current_data.shape)
    # print(current_label.shape)
    # indices = (current_label == 2) | (current_label == 4) | \
    #           (current_label == 8) | (current_label == 12) | \
    #           (current_label == 14) | (current_label == 21) | \
    #           (current_label == 22) | (current_label == 23) | \
    #           (current_label == 30) | (current_label == 35)
    # indices = (current_label == 8)
    # current_class_data = current_data[np.squeeze(indices), :, :]
    # current_class_label = current_label[np.squeeze(indices), :]
    # current_class_label[np.squeeze(current_class_label == 2), :] = 0
    # current_class_label[np.squeeze(current_class_label == 4), :] = 1
    # current_class_label[np.squeeze(current_class_label == 8), :] = 2
    # current_class_label[np.squeeze(current_class_label == 12), :] = 3
    # current_class_label[np.squeeze(current_class_label == 14), :] = 4
    # current_class_label[np.squeeze(current_class_label == 21), :] = 5
    # current_class_label[np.squeeze(current_class_label == 22), :] = 6
    # current_class_label[np.squeeze(current_class_label == 23), :] = 7
    # current_class_label[np.squeeze(current_class_label == 30), :] = 8
    # current_class_label[np.squeeze(current_class_label == 35), :] = 9
    # print(current_data.shape)
    # print(current_label.shape)
    data.append(current_data)
    label.append(current_label)
  return data, label


def loadDataFile_class(filename, class_indices):
    class_data = []
    class_label = []
    current_data, current_label = load_h5(filename)
    label = 0
    for class_index in class_indices:
        indices = (current_label == class_index)
        current_class_data = current_data[np.squeeze(indices), :, :]
        # current_class_label = current_label[np.squeeze(indices), :]
        current_class_label = np.ones((current_class_data.shape[0], 1), np.int32) * label
        label = label + 1
        class_data.append(current_class_data)
        class_label.append(current_class_label)
    class_data = np.concatenate(class_data, axis=0)
    class_label = np.concatenate(class_label, axis=0)
    return class_data, class_label


def loadDataFile_list_all_with_normal(filename_list):
    data = []
    label = []
    for filename in filename_list:
        f = h5py.File(filename)
        current_data = np.concatenate((f['data'][:], f['normal'][:]), axis=-1)
        current_label = f['label'][:]
        data.append(current_data)
        label.append(current_label)
    return data, label


def shuffle_data_2(data, vectors):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,... numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(data.shape[0])
  np.random.shuffle(idx)
  return data[idx, ...], vectors[idx, ...], idx

##########################################

# usuless


# def get_rotation_matrix_to_origin(rotation_vector):
#   cosval_x = np.cos(rotation_vector[0])
#   sinval_x = np.sin(rotation_vector[0])
#   cosval_y = np.cos(rotation_vector[1])
#   sinval_y = np.sin(rotation_vector[1])
#   cosval_z = np.cos(rotation_vector[2])
#   sinval_z = np.sin(rotation_vector[2])
#   rotation_matrix_x = np.array([[1, 0, 0],
#                                 [0, cosval_x, -sinval_x],
#                                 [0, sinval_x, cosval_x]])
#   rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
#                                 [0, 1, 0],
#                                 [-sinval_y, 0, cosval_y]])
#   rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
#                                 [sinval_z, cosval_z, 0],
#                                 [0, 0, 1]])
#   rotation_matrix = np.dot(rotation_matrix_z, rotation_matrix_y)
#   rotation_matrix = np.dot(rotation_matrix, rotation_matrix_x)
#   return rotation_matrix
#
# def get_rotation_vectors_24():
#   rotation_vectors_all = np.zeros((0, 3), dtype=np.float32)
#   for i in range(4):
#     for k in range(4):
#       rotation_vector = np.array([i * math.pi / 2, 0, k * math.pi / 2], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#   for i in range(4):
#     for j in range(2):
#       rotation_vector = np.array([i * math.pi/2, j * math.pi + math.pi / 2, 0], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#   return rotation_vectors_all
#
#
# def get_rotation_vectors_56():
#   rotation_vectors_all = np.zeros((0, 3), dtype=np.float32)
#   for i in range(4):
#     for k in range(4):
#       rotation_vector = np.array([i * math.pi / 2, 0, k * math.pi / 2], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#       rotation_vector = np.array([i * math.pi / 2, math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#       rotation_vector = np.array([i * math.pi / 2, 2 * math.pi -math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#   for i in range(4):
#     for j in range(2):
#       rotation_vector = np.array([i * math.pi/2, j * math.pi + math.pi / 2, 0], dtype=np.float32)
#       rotation_vectors_all = np.vstack((rotation_vectors_all, rotation_vector))
#   return rotation_vectors_all
#
#
# def get_rotation_quaternions_24():
#   rotation_matrics_all = []
#   for i in range(4):
#     for k in range(4):
#       rotation_vector = np.array([i * math.pi / 2, 0, k * math.pi / 2], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   for i in range(4):
#     for j in range(2):
#       rotation_vector = np.array([i * math.pi/2, j * math.pi + math.pi / 2, 0], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   rotation_matrics_all = np.stack(rotation_matrics_all, axis=0)
#   rotation_quaternions_all = rotation_matrics_to_quaternions(rotation_matrics_all)
#   return rotation_quaternions_all
#
#
# def get_rotation_quaternions_56():
#   rotation_matrics_all = []
#   for i in range(4):
#     for k in range(4):
#       rotation_vector = np.array([i * math.pi / 2, 0, k * math.pi / 2], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#       rotation_vector = np.array([i * math.pi / 2, math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#       rotation_vector = np.array([i * math.pi / 2, 2 * math.pi -math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   for i in range(4):
#     for j in range(2):
#       rotation_vector = np.array([i * math.pi/2, j * math.pi + math.pi / 2, 0], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   rotation_matrics_all = np.stack(rotation_matrics_all, axis=0)
#   rotation_quaternions_all = rotation_matrics_to_quaternions(rotation_matrics_all)
#   return rotation_quaternions_all
#
#
# def get_rotation_matrics_56():
#   rotation_matrics_all = []
#   for i in range(4):
#     for k in range(4):
#       rotation_vector = np.array([i * math.pi / 2, 0, k * math.pi / 2], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#       rotation_vector = np.array([i * math.pi / 2, math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#       rotation_vector = np.array([i * math.pi / 2, 2 * math.pi -math.pi / 6, k * math.pi / 2 + math.pi / 4], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   for i in range(4):
#     for j in range(2):
#       rotation_vector = np.array([i * math.pi/2, j * math.pi + math.pi / 2, 0], dtype=np.float32)
#       rotation_matrix = get_rotation_matrix(rotation_vector)
#       rotation_matrics_all.append(rotation_matrix.copy())
#
#   rotation_matrics_all = np.stack(rotation_matrics_all, axis=0)
#   # rotation_quaternions_all = rotation_matrics_to_quaternions(rotation_matrics_all)
#   rotation_matrics_all = np.reshape(rotation_matrics_all, [56, 9])
#   return rotation_matrics_all
#
#
# # rotation classification
# def rotate_data_56_classification(current_data, rotation_vectors_all):
#   try_num = 1
#   current_rotation_class = []
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       rotation_index = np.floor(np.random.uniform() * rotation_vectors_all.shape[0]).astype(np.int32)
#       current_rotation_class.append(rotation_index)
#       rotated_data = np.dot(current_data[shape_index], get_rotation_matrix(rotation_vectors_all[rotation_index]))
#       current_rotated_data.append(rotated_data)
#   current_rotation_class = np.stack(current_rotation_class, axis=0)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   # print(current_rotation_class.shape)
#   # print(current_rotated_data.shape)
#   return current_rotated_data, current_rotation_class
#
#
# def rotate_data_normal_classification(current_data, rotation_vectors_all):
#   try_num = 1
#   current_rotation_class = []
#   current_rotated_data = np.zeros((current_data.shape[0]*try_num, 2048, 3), dtype=np.float32)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       x1, x2, x3 = np.random.rand(3)
#       R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
#                      [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
#                      [0, 0, 1]])
#       v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sqrt(1 - x3)]])
#       H = np.eye(3) - 2 * v * v.T
#       M = -H * R
#       rotated_data = np.dot(current_data[shape_index], M)
#       # print(rotated_data.shape)
#       #current_rotated_data.append(rotated_data)
#       current_rotated_data[shape_index*try_num+index, :, :] = rotated_data.copy()
#
#       M_expand = np.tile(np.expand_dims(M, axis=0), (rotation_vectors_all.shape[0], 1, 1))
#       M_distance = np.sum(np.sum(np.power(M_expand - get_rotation_matrics(rotation_vectors_all), 2), axis=2), axis=1)
#       rotation_index = np.argmin(M_distance)
#
#       # min_distance = 1e6
#       # min_i = 0
#       # for i in range(rotation_vectors_all.shape[0]):
#       #   if np.linalg.norm(get_rotation_matrix(rotation_vectors_all[i]) - M) < min_distance:
#       #     min_i = i
#       #     min_distance = np.linalg.norm(get_rotation_matrix(rotation_vectors_all[i]) - M)
#       # rotation_index = min_i
#
#       current_rotation_class.append(rotation_index)
#   current_rotation_class = np.stack(current_rotation_class, axis=0)
#   # current_rotated_data = np.stack(current_rotated_data, axis=0)
#   # print(current_rotation_class.shape)
#   # print(current_rotated_data.shape)
#   return current_rotated_data, current_rotation_class
#
#
# def rotate_data_to_origin_classification(current_data, rotation_vectors_all, current_rotation_class):
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     rotation_index = current_rotation_class[shape_index]
#     rotated_data = np.dot(current_data[shape_index], get_rotation_matrix_to_origin(rotation_vectors_all[rotation_index]))
#     current_rotated_data.append(rotated_data)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data
#
#
# def rotate_data_to_origin_classification_2(current_data, current_rotation_class):
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     rotation_index = current_rotation_class[shape_index]
#     rotation_x_index = rotation_index % 18
#     rotation_z_index = rotation_index // 18
#     rotation_angle_x = rotation_x_index * (np.pi / 9)
#     rotation_angle_z = rotation_z_index * (np.pi / 9)
#     cosval_x = np.cos(rotation_angle_x)
#     sinval_x = np.sin(rotation_angle_x)
#     cosval_z = np.cos(rotation_angle_z)
#     sinval_z = np.sin(rotation_angle_z)
#     rotation_matrix_x = np.array([[1, 0, 0],
#                                   [0, cosval_x, -sinval_x],
#                                   [0, sinval_x, cosval_x]])
#     rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
#                                   [sinval_z, cosval_z, 0],
#                                   [0, 0, 1]])
#     rotation_matrix = np.dot(rotation_matrix_z, rotation_matrix_x)
#     rotated_data = np.dot(current_data[shape_index], rotation_matrix)
#     current_rotated_data.append(rotated_data)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data
#
#
# def rotate_data_to_origin_classification_3(current_data, current_rotation_class):
#   current_rotated_data = np.zeros((current_data.shape[0], current_data.shape[1], 3), dtype=np.float32)
#   for shape_index in range(current_data.shape[0]):
#     rotation_index = current_rotation_class[shape_index]
#     rotation_x_index = rotation_index % 18
#     rotation_z_index = rotation_index // 18
#     rotation_angle_x = rotation_x_index * 0.05
#     rotation_angle_z = rotation_z_index * 0.05
#
#     v = np.matrix([[np.cos(2 * np.pi * rotation_angle_x) * np.sqrt(rotation_angle_z)],
#                    [np.sqrt(1 - rotation_angle_z)],
#                    [np.sin(2 * np.pi * rotation_angle_x) * np.sqrt(rotation_angle_z)]])
#     H = np.eye(3) - 2 * v * v.T
#
#     rotation_matrix = -H
#     rotated_data = np.dot(current_data[shape_index], rotation_matrix)
#     current_rotated_data[shape_index, :, :] = rotated_data.copy()
#   # current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data

# rotation regression
# def rotate_data_56_vectors_regression(current_data, rotation_vectors_all):
#   try_num = 1
#   current_rotation_vectors = []
#   current_rotated_data = []
#   rotation_acc = 0
#   rotation_indices = np.arange(rotation_vectors_all.shape[0])
#   np.random.shuffle(rotation_indices)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       rotation_index = rotation_indices[rotation_acc]
#       rotation_acc += 1
#       if rotation_acc == rotation_vectors_all.shape[0]:
#           rotation_indices = np.arange(rotation_vectors_all.shape[0])
#           np.random.shuffle(rotation_indices)
#           rotation_acc = 0
#       current_rotation_vectors.append(rotation_vectors_all[rotation_index])
#       rotated_data = np.dot(current_data[shape_index], get_rotation_matrix(rotation_vectors_all[rotation_index]))
#       current_rotated_data.append(rotated_data)
#   current_rotation_vectors = np.stack(current_rotation_vectors, axis=0)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data, current_rotation_vectors
#
#
# def rotate_data_normal_vectors_regression(current_data):
#   try_num = 1
#   current_rotated_data = np.zeros((current_data.shape[0]*try_num, 2048, 3), dtype=np.float32)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       x1, x2, x3 = np.random.rand(3)
#       R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
#                      [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
#                      [0, 0, 1]])
#       v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sqrt(1 - x3)]])
#       H = np.eye(3) - 2 * v * v.T
#       M = -H * R
#       rotated_data = np.dot(current_data[shape_index], M)
#       # print(rotated_data.shape)
#       #current_rotated_data.append(rotated_data)
#       current_rotated_data[shape_index*try_num+index, :, :] = rotated_data.copy()
#   return current_rotated_data
#
#
# def rotate_data_to_origin_vectors_regression(current_data, current_rotation_vectors):
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     rotation_vector = current_rotation_vectors[shape_index]
#     rotated_data = np.dot(current_data[shape_index], get_rotation_matrix_to_origin(rotation_vector))
#     current_rotated_data.append(rotated_data)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data
#
#
# def rotate_data_56_quaternions_regression(current_data, rotation_vectors_all):
#   try_num = 1
#   current_rotation_vectors = []
#   current_rotated_data = []
#   rotation_acc = 0
#   rotation_indices = np.arange(rotation_vectors_all.shape[0])
#   np.random.shuffle(rotation_indices)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       rotation_index = rotation_indices[rotation_acc]
#       rotation_acc += 1
#       if rotation_acc == rotation_vectors_all.shape[0]:
#         rotation_indices = np.arange(rotation_vectors_all.shape[0])
#         np.random.shuffle(rotation_indices)
#         rotation_acc = 0
#       current_rotation_vectors.append(rotation_vectors_all[rotation_index])
#       rotated_data = np.dot(current_data[shape_index], quaternion_to_rotation_matrix(rotation_vectors_all[rotation_index]))
#       current_rotated_data.append(rotated_data)
#   current_rotation_vectors = np.stack(current_rotation_vectors, axis=0)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data, current_rotation_vectors
#
#
# def rotate_data_normal_quaternions_regression(current_data):
#   try_num = 1
#   current_rotation_vectors = []
#   current_rotated_data = np.zeros((current_data.shape[0]*try_num, 2048, 3), dtype=np.float32)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       # rotation_angle = np.random.uniform() * 2 * np.pi
#       # cosval = np.cos(rotation_angle)
#       # sinval = np.sin(rotation_angle)
#       # M = np.array([[cosval, 0, sinval],
#       #                [0, 1, 0],
#       #                [-sinval, 0, cosval]])
#
#       x1, x2, x3 = np.random.rand(3)
#       R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
#                      [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
#                      [0, 0, 1]])
#       v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sqrt(1 - x3)]])
#       H = np.eye(3) - 2 * v * v.T
#       M = -H * R
#
#       rotated_data = np.dot(current_data[shape_index], M)
#       # print(rotated_data.shape)
#       #current_rotated_data.append(rotated_data)
#       current_rotated_data[shape_index*try_num+index, :, :] = rotated_data.copy()
#       current_rotation_vectors.append(rotation_matrix_to_quaternion(M))
#   current_rotation_vectors = np.stack(current_rotation_vectors, axis=0)
#   return current_rotated_data, current_rotation_vectors
#
#
# def rotate_data_to_origin_quaternions_regression(current_data, current_rotation_vectors):
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     rotation_vector = current_rotation_vectors[shape_index].copy()
#     rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
#     rotation_vector[1:] = -rotation_vector[1:]
#     rotated_data = np.dot(current_data[shape_index], quaternion_to_rotation_matrix(rotation_vector))
#     current_rotated_data.append(rotated_data)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data
#
#
# def rotate_data_56_matrics_regression(current_data, rotation_vectors_all):
#   try_num = 1
#   current_rotation_vectors = []
#   current_rotated_data = []
#   rotation_acc = 0
#   rotation_indices = np.arange(rotation_vectors_all.shape[0])
#   np.random.shuffle(rotation_indices)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       rotation_index = rotation_indices[rotation_acc]
#       rotation_acc += 1
#       if rotation_acc == rotation_vectors_all.shape[0]:
#         rotation_indices = np.arange(rotation_vectors_all.shape[0])
#         np.random.shuffle(rotation_indices)
#         rotation_acc = 0
#       current_rotation_vectors.append(rotation_vectors_all[rotation_index])
#       rotated_data = np.dot(current_data[shape_index], np.reshape(rotation_vectors_all[rotation_index], [3, 3]))
#       current_rotated_data.append(rotated_data)
#   current_rotation_vectors = np.stack(current_rotation_vectors, axis=0)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data, current_rotation_vectors
#
#
# def rotate_data_normal_matrics_regression(current_data):
#   try_num = 1
#   current_rotation_vectors = []
#   current_rotated_data = np.zeros((current_data.shape[0]*try_num, 2048, 3), dtype=np.float32)
#   for shape_index in range(current_data.shape[0]):
#     for index in range(try_num):
#       # rotation_angle = np.random.uniform() * 2 * np.pi
#       # cosval = np.cos(rotation_angle)
#       # sinval = np.sin(rotation_angle)
#       # M = np.array([[cosval, 0, sinval],
#       #                [0, 1, 0],
#       #                [-sinval, 0, cosval]])
#
#       x1, x2, x3 = np.random.rand(3)
#       R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
#                      [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
#                      [0, 0, 1]])
#       v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
#                      [np.sqrt(1 - x3)]])
#       H = np.eye(3) - 2 * v * v.T
#       M = -H * R
#
#       rotated_data = np.dot(current_data[shape_index], M)
#       # print(rotated_data.shape)
#       #current_rotated_data.append(rotated_data)
#       current_rotated_data[shape_index*try_num+index, :, :] = rotated_data.copy()
#       current_rotation_vectors.append(np.reshape(M, [9]))
#   current_rotation_vectors = np.stack(current_rotation_vectors, axis=0)
#   return current_rotated_data, current_rotation_vectors
#
#
# def rotate_data_to_origin_matrics_regression(current_data, current_rotation_vectors):
#   current_rotated_data = []
#   for shape_index in range(current_data.shape[0]):
#     rotation_vector = current_rotation_vectors[shape_index].copy()
#     rotation_matrix = np.reshape(rotation_vector, [3, 3])
#     rotation_matrix[0:1, :] = rotation_matrix[0:1, :] / np.linalg.norm(rotation_matrix[0:1, :])
#     rotation_matrix[1:2, :] = rotation_matrix[1:2, :] / np.linalg.norm(rotation_matrix[1:2, :])
#     rotation_matrix[2:3, :] = rotation_matrix[2:3, :] / np.linalg.norm(rotation_matrix[2:3, :])
#     rotated_data = np.dot(current_data[shape_index], rotation_matrix.transpose())
#     current_rotated_data.append(rotated_data)
#   current_rotated_data = np.stack(current_rotated_data, axis=0)
#   return current_rotated_data

