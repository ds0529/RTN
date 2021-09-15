import tensorflow as tf
import numpy as np
from scipy import spatial
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data_path', default='/mnt/dengshuang/data/modelnet40/aligned_modelnet40_ply_hdf5_2048', help='data path')
parser.add_argument('--model', default='learning_rotation_nets_mix_model', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='result/log_learning_rotation_mix_model_so3/best_model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='result/dump_learning_rotation_mix_model_so3', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()
angle_cell = np.pi / 6
part = 0
neighbors = 1
plane = 0


DATA_PATH = FLAGS.data_path
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
TRANSFORMER_MODEL = MODEL
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(DATA_PATH+'/shape_names.txt')]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    DATA_PATH+'/train_files.txt')
TEST_FILES = provider.getDataFiles(\
    DATA_PATH+'/test_files.txt')
class_indices = [0, 2, 7, 17, 20, 22, 24, 25, 30, 35]
# class_indices = [0, 2, 20, 22, 25, 30, 35]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # transformer
        # rotation_class_num, rotation_matrics_all, rotation_y_all = provider.get_z_rotation_matrics_all()
        rotation_class_num, rotation_matrics_all, rotation_plane_all, rotation_matrics_sphere_all, rotation_y_all = provider.get_so3_rotation_matrics_all()
        with tf.variable_scope('learning_rotation') as sc:
            pointclouds_origin_pl, rotation_class_pl = TRANSFORMER_MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_transformer_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred_rotation_class, transformer_end_points = TRANSFORMER_MODEL.get_model(pointclouds_origin_pl, rotation_class_num, is_training_transformer_pl)
            transformer_loss = TRANSFORMER_MODEL.get_loss(pred_rotation_class, rotation_class_pl)

            correct = tf.equal(tf.argmax(pred_rotation_class, 1), tf.to_int64(rotation_class_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

        # classifier
        DGCNN_MODEL = importlib.import_module('dgcnn')
        with tf.variable_scope('dgcnn') as sc:
            is_training_classifier_pl = tf.placeholder(tf.bool, shape=())

            # simple model
            pred, end_points, _ = DGCNN_MODEL.get_model(pointclouds_origin_pl, is_training_classifier_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    # saver.restore(sess, MODEL_PATH)

    restore_into_scope(MODEL_PATH, 'learning_rotation', sess)
    dgcnn_model_path = 'result/log_so3_rotation/best_model.ckpt'
    restore_into_scope(dgcnn_model_path, 'dgcnn', sess)
    log_string("Model restored.")

    ops = {'rotation_matrics_all': rotation_matrics_all,
           'rotation_plane_all': rotation_plane_all,
           'rotation_matrics_sphere_all': rotation_matrics_sphere_all,
           'rotation_y_all': rotation_y_all,
           'pointclouds_origin_pl': pointclouds_origin_pl,
           'sampled_points': transformer_end_points['sampled_points'],
           'rotation_class_pl': rotation_class_pl,
           'is_training_transformer_pl':is_training_transformer_pl,
           'pred_rotation_class': pred_rotation_class,
           'transformer_loss': transformer_loss,
           'accuracy': accuracy,
           'is_training_classifier_pl': is_training_classifier_pl,
           'dgcnn_transformed_points': end_points['transformed_points']
           }

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training_transformer = False
    try_num = 10
    acc_list = np.zeros((try_num), np.float32)
    dist_list = np.zeros((try_num), np.float32)
    disr_list = np.zeros((try_num), np.float32)
    for t in range(try_num):
        total_correct = 0
        total_seen = 0
        dis_transform_all = 0.0
        dis_rotated_all = 0.0
        for fn in range(len(TEST_FILES)):
            log_string('----'+str(fn)+'----')
            if part:
                current_data, current_label = provider.loadDataFile_class(DATA_PATH+TEST_FILES[fn], class_indices)
            else:
                current_data, current_label = provider.loadDataFile(DATA_PATH+TEST_FILES[fn])


            current_data = current_data[:,0:NUM_POINT,:]
            print(current_data.shape)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            print(file_size)

            # transform_list = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                cur_batch_size = end_idx - start_idx

                rotated_data, current_rotation_matrix, current_rotation_vector = provider.rotate_point_cloud_by_so3(current_data[start_idx:end_idx, :, :])
                # current_rotation_class = provider.get_z_rotation_class(current_rotation_vector, ops['rotation_y_all'])
                current_rotation_class = provider.get_so3_rotation_class(current_rotation_matrix,
                                                                         ops['rotation_matrics_all'],
                                                                         ops['rotation_plane_all'],
                                                                         ops['rotation_matrics_sphere_all'],
                                                                         ops['rotation_y_all'])

                # get rotation class
                feed_dict = {ops['pointclouds_origin_pl']: rotated_data,
                             ops['is_training_transformer_pl']: is_training_transformer}
                pred_rotation_class, sampled_points = sess.run([ops['pred_rotation_class'], ops['sampled_points']], feed_dict=feed_dict)
                transformed_points = provider.rotate_data_to_origin_classification_so3(rotated_data,
                                                                                       ops['rotation_matrics_all'],
                                                                                       np.argmax(pred_rotation_class, axis=1))

                dis_transform = chamfer_distance_sklearn(current_data[start_idx:end_idx, :, :], transformed_points)
                dis_rotated = chamfer_distance_sklearn(current_data[start_idx:end_idx, :, :], rotated_data)
                dis_transform_all += dis_transform
                dis_rotated_all += dis_rotated
                pred_val = np.argmax(pred_rotation_class, 1)

                if neighbors:
                    failed_pred_val = []
                    failed_current_rotation_class = []
                    failed_rotated_data = []
                    failed_transformed_points = []

                    y_num = ops['rotation_plane_all'].shape[0]
                    longitude_num = int(2 * np.pi / angle_cell)
                    sphere_num = ops['rotation_y_all'].shape[0]

                    sphere_pred = (pred_val / y_num).astype(np.int32)
                    y_pred = pred_val % y_num
                    sphere_gt = (current_rotation_class / y_num).astype(np.int32)
                    y_gt = current_rotation_class % y_num

                    for shape_index in range(sphere_pred.shape[0]):
                        sphere_success = 0
                        success = 0
                        if sphere_gt[shape_index] == sphere_pred[shape_index]: sphere_success = 1
                        if sphere_pred[shape_index] == 0:
                            if sphere_gt[shape_index] > 0 and sphere_gt[shape_index] <= longitude_num: sphere_success = 1
                        elif sphere_pred[shape_index] > 0 and sphere_pred[shape_index] <= longitude_num:
                            if sphere_gt[shape_index] == 0: sphere_success = 1
                            if sphere_gt[shape_index] == sphere_pred[shape_index] + longitude_num: sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 1:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + longitude_num - 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - 1: sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 0:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - longitude_num + 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + 1: sphere_success = 1
                        elif sphere_pred[shape_index] > longitude_num and sphere_pred[shape_index] < (sphere_num - longitude_num - 1):
                            if sphere_gt[shape_index] == sphere_pred[shape_index] - longitude_num: sphere_success = 1
                            if sphere_gt[shape_index] == sphere_pred[shape_index] + longitude_num: sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 1:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + longitude_num - 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - 1: sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 0:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - longitude_num + 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + 1: sphere_success = 1
                        elif sphere_pred[shape_index] >= (sphere_num - longitude_num - 1) and sphere_pred[shape_index] < (sphere_num - 1):
                            if sphere_gt[shape_index] == sphere_pred[shape_index] - longitude_num: sphere_success = 1
                            if sphere_gt[shape_index] == (sphere_num - 1): sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 1:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + longitude_num - 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - 1: sphere_success = 1
                            if sphere_pred[shape_index] % longitude_num == 0:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] - longitude_num + 1: sphere_success = 1
                            else:
                                if sphere_gt[shape_index] == sphere_pred[shape_index] + 1: sphere_success = 1
                        elif sphere_pred[shape_index] == (sphere_num - 1):
                            if sphere_gt[shape_index] >= (sphere_num - longitude_num - 1) and sphere_gt[shape_index] < (sphere_num - 1): sphere_success = 1

                        if sphere_success == 1:
                            if plane:
                                success = 1
                                total_correct += 1
                            else:
                                if y_pred[shape_index] == y_gt[shape_index]:
                                    success = 1
                                    total_correct += 1
                                elif y_pred[shape_index] == 0:
                                    if y_gt[shape_index] == y_num - 1 or y_gt[shape_index] == 1:
                                        success = 1
                                        total_correct += 1
                                elif y_pred[shape_index] == y_num - 1:
                                    if y_gt[shape_index] == y_num - 2 or y_gt[shape_index] == 0:
                                        success = 1
                                        total_correct += 1
                                elif y_pred[shape_index] == (y_gt[shape_index] + 1) or y_pred[shape_index] == (y_gt[shape_index] - 1):
                                    success = 1
                                    total_correct += 1

                        if success == 0:
                            failed_pred_val.append(pred_val[shape_index])
                            failed_current_rotation_class.append(current_rotation_class[shape_index])
                            failed_rotated_data.append(rotated_data[shape_index])
                            failed_transformed_points.append(transformed_points[shape_index])

                    if len(failed_rotated_data) > 0:
                        failed_pred_val = np.stack(failed_pred_val, axis=0)
                        failed_current_rotation_class = np.stack(failed_current_rotation_class, axis=0)
                        failed_rotated_data = np.stack(failed_rotated_data, axis=0)
                        failed_transformed_points = np.stack(failed_transformed_points, axis=0)
                else:
                    if plane:
                        y_num = ops['rotation_plane_all'].shape[0]
                        longitude_num = int(2 * np.pi / angle_cell)
                        sphere_num = ops['rotation_y_all'].shape[0]

                        sphere_pred = (pred_val / y_num).astype(np.int32)
                        y_pred = pred_val % y_num
                        sphere_gt = (current_rotation_class / y_num).astype(np.int32)
                        y_gt = current_rotation_class % y_num

                        total_correct += np.sum(sphere_pred == sphere_gt)
                    else:
                        total_correct += np.sum(pred_val == current_rotation_class)

                total_seen += cur_batch_size

        log_string('transform distance: %f' % (dis_transform_all / float(total_seen)))
        log_string('rotated distance: %f' % (dis_rotated_all / float(total_seen)))
        log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        dist_list[t] = dis_transform_all / float(total_seen)
        disr_list[t] = dis_rotated_all / float(total_seen)
        acc_list[t] = total_correct / float(total_seen)
    mean_dist = dist_list.mean()
    mean_disr = disr_list.mean()
    mean_acc = acc_list.mean()
    std_acc = acc_list.std()
    log_string('eval mean transform distance: %f' % (mean_dist))
    log_string('eval mean rotated distance: %f' % (mean_disr))
    log_string('eval mean accuracy: %f' % (mean_acc))
    log_string('eval std accuracy: %f' % (std_acc))


def restore_into_scope(model_path, scope_name, sess):
    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + '/')]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        tensor_name = tensor_name[0:-2] # remove ':0'
        tensor_name = tensor_name.replace(scope_name + '/', '') #remove scope
        load_dict.update({tensor_name: tensors_to_load[j]})
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    log_string("Model restored from: {0} into scope: {1}.".format(model_path, scope_name))


def chamfer_distance_sklearn(array1, array2):
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = spatial.KDTree(array1[i], leafsize=num_point+1)
        tree2 = spatial.KDTree(array2[i], leafsize=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/2
    return dist



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=12)
    LOG_FOUT.close()
