import argparse
import math
import h5py
import numpy as np
import os
import sys
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
# import provider


def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)


def loadDataFile_list_all(filename_list):
  data = []
  label = []
  for filename in filename_list:
    current_data, current_label = load_h5(filename)
    data.append(current_data)
    label.append(current_label)
  return data, label


PLY_DIR = os.path.join(BASE_DIR, 'ply')
ply_dict = {}
ply_dict['save_ply_dir'] = PLY_DIR
if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
ply_dict['row_index'] = 0
ply_dict['col_index'] = 0
ply_dict['index'] = 0
ply_dict['gap'] = 2
ply_dict['max_index'] = 10
ply_dict['vertices_list'] = []
ply_dict['ply_index'] = 0
def create_output(points1):
    shape_num = points1.shape[0]
    for i in range(shape_num):
        point_num = points1.shape[1]
        colors1 = np.tile(np.array([0, 0, 0], np.int)*255, [point_num, 1])
        colors1[:, 0] = colors1[:, 0] + (points1[i, :, 2] + 1) / 2 * 255
        colors1[:, 1] = colors1[:, 1] + (points1[i, :, 2] + 1) / 2 * 255
        colors1[:, 2] = colors1[:, 2] + (points1[i, :, 2]+1)/2*255

        points = np.vstack([points1[i]+np.array([ply_dict['row_index']*ply_dict['gap'], ply_dict['col_index']*ply_dict['gap'], 0])])
        colors = np.vstack([colors1])

        vertices = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])

        if ply_dict['col_index'] < ply_dict['max_index'] -1:
            ply_dict['vertices_list'].append(vertices)
            ply_dict['col_index'] = ply_dict['col_index'] + 1
            ply_dict['index'] += 1
            if i == shape_num - 1:
                vertices_all = np.vstack(ply_dict['vertices_list'])
                if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
                filename = ply_dict['save_ply_dir'] + '/' + str(ply_dict['ply_index']) + '.ply'
                np.savetxt(filename, vertices_all, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
                ply_header = '''ply
                                		format ascii 1.0
                                		element vertex %d
                                		property float x
                                		property float y
                                		property float z
                                		property uchar red
                                		property uchar green
                                		property uchar blue
                                		end_header
                                		\n
                                		''' % (ply_dict['index'] * point_num)
                with open(filename, 'r+') as f:
                    old = f.read()
                    f.seek(0)
                    f.write(ply_header % dict(vert_num=len(vertices)))
                    f.write(old)
                ply_dict['vertices_list'] = []
                ply_dict['col_index'] = 0
                ply_dict['row_index'] = 0
                ply_dict['index'] = 0
                ply_dict['ply_index'] = ply_dict['ply_index'] + 1
        elif ply_dict['row_index'] < ply_dict['max_index'] -1:
            ply_dict['vertices_list'].append(vertices)
            ply_dict['col_index'] = 0
            ply_dict['row_index'] = ply_dict['row_index'] + 1
            ply_dict['index'] += 1
            if i == shape_num - 1:
                vertices_all = np.vstack(ply_dict['vertices_list'])
                if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
                filename = ply_dict['save_ply_dir'] + '/' + str(ply_dict['ply_index']) + '.ply'
                np.savetxt(filename, vertices_all, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
                ply_header = '''ply
                                            		format ascii 1.0
                                            		element vertex %d
                                            		property float x
                                            		property float y
                                            		property float z
                                            		property uchar red
                                            		property uchar green
                                            		property uchar blue
                                            		end_header
                                            		\n
                                            		''' % (
                        ply_dict['index'] * point_num)
                with open(filename, 'r+') as f:
                    old = f.read()
                    f.seek(0)
                    f.write(ply_header % dict(vert_num=len(vertices)))
                    f.write(old)
                ply_dict['vertices_list'] = []
                ply_dict['col_index'] = 0
                ply_dict['row_index'] = 0
                ply_dict['index'] = 0
                ply_dict['ply_index'] = ply_dict['ply_index'] + 1
        else:
            ply_dict['vertices_list'].append(vertices)
            ply_dict['index'] += 1
            vertices_all = np.vstack(ply_dict['vertices_list'])
            if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
            filename = ply_dict['save_ply_dir'] + '/' + str(ply_dict['ply_index']) + '.ply'
            np.savetxt(filename, vertices_all, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
            ply_header = '''ply
                		format ascii 1.0
                		element vertex %d
                		property float x
                		property float y
                		property float z
                		property uchar red
                		property uchar green
                		property uchar blue
                		end_header
                		\n
                		''' % (ply_dict['index']*point_num)
            with open(filename, 'r+') as f:
                old = f.read()
                f.seek(0)
                f.write(ply_header % dict(vert_num=len(vertices)))
                f.write(old)
            ply_dict['vertices_list'] = []
            ply_dict['col_index'] = 0
            ply_dict['row_index'] = 0
            ply_dict['index'] = 0
            ply_dict['ply_index'] = ply_dict['ply_index']+1


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


# ModelNet40 official train/test split
TRAIN_FILES = getDataFiles( \
    'modelnet40_ply_hdf5_2048/train_files.txt')
TEST_FILES = getDataFiles( \
    'modelnet40_ply_hdf5_2048/test_files.txt')
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(BASE_DIR, '/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = provider.getDataFiles(\
#     os.path.join(BASE_DIR, '/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048/test_files.txt'))

# get class data
data, label = loadDataFile_list_all(TRAIN_FILES)
test_data, test_label = loadDataFile_list_all(TEST_FILES)

# prepare trainset
print('save file')
file_num = len(data)
FILE_PLY_DIR = os.path.join(PLY_DIR, 'file')
if not os.path.exists(FILE_PLY_DIR): os.mkdir(FILE_PLY_DIR)
for f in range(file_num):
    print(data[f].shape)
    ply_dict['save_ply_dir'] = os.path.join(FILE_PLY_DIR, 'f_'+str(f))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(data[f])
    # pdb.set_trace()

print('save class and class_aligned')
NUM_CLASS = 40
CLASS_PLY_DIR = os.path.join(PLY_DIR, 'class')
if not os.path.exists(CLASS_PLY_DIR): os.mkdir(CLASS_PLY_DIR)
ALIGN_ANGLES_DIR = 'data_align_angles'
CLASS_ROTATED_PLY_DIR = os.path.join(PLY_DIR, 'class_aligned')
if not os.path.exists(CLASS_ROTATED_PLY_DIR): os.mkdir(CLASS_ROTATED_PLY_DIR)
class_indices = []
align_angles = []
for f in range(file_num):
    class_indices.append(np.zeros((data[f].shape[0], 2), dtype=np.int))
for c in range(0, NUM_CLASS):
    class_data = []
    class_num = 0
    for f in range(file_num):
        for i in range(data[f].shape[0]):
            if label[f][i] == c:
                class_data.append(data[f][i])
                class_indices[f][i, 0] = c
                class_indices[f][i, 1] = class_num
                class_num += 1
    class_data = np.stack(class_data, axis=0)
    print(str(c)+':')
    print(class_data.shape)
    ply_dict['save_ply_dir'] = os.path.join(CLASS_PLY_DIR, 'c_'+str(c))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(class_data)
    # pdb.set_trace()

    class_align_angles = np.loadtxt(os.path.join(ALIGN_ANGLES_DIR, 'c_%d.txt'%c), delimiter=',')/180*np.pi
    align_angles.append(class_align_angles)
    rotated_class_data = []
    for i in range(class_num):
        rotation_matrix = get_rotation_matrix(class_align_angles[i])
        rotated_class_data.append(np.dot(class_data[i], rotation_matrix))
    rotated_class_data = np.stack(rotated_class_data, axis=0)
    ply_dict['save_ply_dir'] = os.path.join(CLASS_ROTATED_PLY_DIR, 'c_' + str(c))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(rotated_class_data)
    # pdb.set_trace()

print('save h5')
ALIGNED_MODELNET_DIR = os.path.join(BASE_DIR, 'aligned_modelnet40_ply_hdf5_2048')
if not os.path.exists(ALIGNED_MODELNET_DIR): os.mkdir(ALIGNED_MODELNET_DIR)
for f in range(file_num):
    print(data[f].shape)
    aligned_file_data = []
    for i in range(data[f].shape[0]):
        align_angle = align_angles[class_indices[f][i, 0]][class_indices[f][i, 1]]
        rotation_matrix = get_rotation_matrix(align_angle)
        aligned_file_data.append(np.dot(data[f][i], rotation_matrix))
    aligned_file_data = np.stack(aligned_file_data, axis=0)
    # Create a new file
    h5f = h5py.File(os.path.join(ALIGNED_MODELNET_DIR, 'ply_data_train%d.h5'%f), 'w')
    h5f.create_dataset('data', data=aligned_file_data)
    h5f.create_dataset('label', data=label[f])
    h5f.close()

print('save file_aligned')
ALIGNED_FILE_PLY_DIR = os.path.join(PLY_DIR, 'file_aligned')
if not os.path.exists(ALIGNED_FILE_PLY_DIR): os.mkdir(ALIGNED_FILE_PLY_DIR)
for f in range(file_num):
    aligned_file_data = h5py.File(os.path.join(ALIGNED_MODELNET_DIR, 'ply_data_train%d.h5'%f))['data'][:]
    print(aligned_file_data.shape)
    ply_dict['save_ply_dir'] = os.path.join(ALIGNED_FILE_PLY_DIR, 'f_'+str(f))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(aligned_file_data)
    # pdb.set_trace()


# prepare testset
print('save test_file')
file_num = len(test_data)
FILE_PLY_DIR = os.path.join(PLY_DIR, 'test_file')
if not os.path.exists(FILE_PLY_DIR): os.mkdir(FILE_PLY_DIR)
for f in range(file_num):
    print(test_data[f].shape)
    ply_dict['save_ply_dir'] = os.path.join(FILE_PLY_DIR, 'f_'+str(f))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(test_data[f])
    # pdb.set_trace()

print('save test_class and test_class_aligned')
NUM_CLASS = 40
CLASS_PLY_DIR = os.path.join(PLY_DIR, 'test_class')
if not os.path.exists(CLASS_PLY_DIR): os.mkdir(CLASS_PLY_DIR)
ALIGN_ANGLES_DIR = 'test_data_align_angles'
CLASS_ROTATED_PLY_DIR = os.path.join(PLY_DIR, 'test_class_aligned')
if not os.path.exists(CLASS_ROTATED_PLY_DIR): os.mkdir(CLASS_ROTATED_PLY_DIR)
class_indices = []
align_angles = []
for f in range(file_num):
    class_indices.append(np.zeros((test_data[f].shape[0], 2), dtype=np.int))
for c in range(0, NUM_CLASS):
    class_data = []
    class_num = 0
    for f in range(file_num):
        for i in range(test_data[f].shape[0]):
            if test_label[f][i] == c:
                class_data.append(test_data[f][i])
                class_indices[f][i, 0] = c
                class_indices[f][i, 1] = class_num
                class_num += 1
    class_data = np.stack(class_data, axis=0)
    print(str(c)+':')
    print(class_data.shape)
    ply_dict['save_ply_dir'] = os.path.join(CLASS_PLY_DIR, 'c_'+str(c))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(class_data)
    # pdb.set_trace()

    class_align_angles = np.loadtxt(os.path.join(ALIGN_ANGLES_DIR, 'c_%d.txt'%c), delimiter=',')/180*np.pi
    align_angles.append(class_align_angles)
    rotated_class_data = []
    for i in range(class_num):
        rotation_matrix = get_rotation_matrix(class_align_angles[i])
        rotated_class_data.append(np.dot(class_data[i], rotation_matrix))
    rotated_class_data = np.stack(rotated_class_data, axis=0)
    ply_dict['save_ply_dir'] = os.path.join(CLASS_ROTATED_PLY_DIR, 'c_' + str(c))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(rotated_class_data)
    # pdb.set_trace()

print('save test h5')
ALIGNED_MODELNET_DIR = os.path.join(BASE_DIR, 'aligned_modelnet40_ply_hdf5_2048')
if not os.path.exists(ALIGNED_MODELNET_DIR): os.mkdir(ALIGNED_MODELNET_DIR)
for f in range(file_num):
    print(test_data[f].shape)
    aligned_file_data = []
    for i in range(test_data[f].shape[0]):
        align_angle = align_angles[class_indices[f][i, 0]][class_indices[f][i, 1]]
        rotation_matrix = get_rotation_matrix(align_angle)
        aligned_file_data.append(np.dot(test_data[f][i], rotation_matrix))
    aligned_file_data = np.stack(aligned_file_data, axis=0)
    # Create a new file
    h5f = h5py.File(os.path.join(ALIGNED_MODELNET_DIR, 'ply_data_test%d.h5'%f), 'w')
    h5f.create_dataset('data', data=aligned_file_data)
    h5f.create_dataset('label', data=test_label[f])
    h5f.close()

print('save test_file_aligned')
ALIGNED_FILE_PLY_DIR = os.path.join(PLY_DIR, 'test_file_aligned')
if not os.path.exists(ALIGNED_FILE_PLY_DIR): os.mkdir(ALIGNED_FILE_PLY_DIR)
for f in range(file_num):
    aligned_file_data = h5py.File(os.path.join(ALIGNED_MODELNET_DIR, 'ply_data_test%d.h5'%f))['data'][:]
    print(aligned_file_data.shape)
    ply_dict['save_ply_dir'] = os.path.join(ALIGNED_FILE_PLY_DIR, 'f_'+str(f))
    if not os.path.exists(ply_dict['save_ply_dir']): os.mkdir(ply_dict['save_ply_dir'])
    ply_dict['ply_index'] = 0
    create_output(aligned_file_data)
    # pdb.set_trace()
