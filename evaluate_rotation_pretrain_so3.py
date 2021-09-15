import tensorflow as tf
import numpy as np
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


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data_path', default='/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048', help='data path')
parser.add_argument('--transformer_model', default='learning_rotation_nets_mix_model', help='Transformer model name: dgcnn [default: dgcnn]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--transformer_model_path', default='result/log_learning_rotation_mix_model_so3/best_model.ckpt', help='transformer model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--model_path', default='result/log_rotation_pretrain_mix_model_dgcnn_so3/best_model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='result/dump_rotation_pretrain_mix_model_dgcnn_so3', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


DATA_PATH = FLAGS.data_path
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
TRANSFORMER_MODEL = importlib.import_module(FLAGS.transformer_model)
DUMP_DIR = FLAGS.dump_dir
ERROR_PLY_DIR = os.path.join(DUMP_DIR, 'error_ply')
TRANSFORM_PLY_DIR = os.path.join(DUMP_DIR, 'transform_ply')
if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
# if not os.path.exists(ERROR_PLY_DIR): os.mkdir(ERROR_PLY_DIR)
# f not os.path.exists(TRANSFORM_PLY_DIR): os.mkdir(TRANSFORM_PLY_DIR)
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


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # transformer
        # rotation_class_num, rotation_matrics_all, rotation_y_all = provider.get_z_rotation_matrics_all()
        rotation_class_num, rotation_matrics_all, rotation_plane_all, rotation_matrics_sphere_all, rotation_y_all = provider.get_so3_rotation_matrics_all()
        with tf.variable_scope('learning_rotation') as sc:
            pointclouds_origin_pl, rotation_class_pl = TRANSFORMER_MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_transformer_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred_rotation_class, _ = TRANSFORMER_MODEL.get_model(pointclouds_origin_pl, rotation_class_num, is_training_transformer_pl)
            transformer_loss = TRANSFORMER_MODEL.get_loss(pred_rotation_class, rotation_class_pl)

        # classifier
        with tf.variable_scope('dgcnn') as sc:
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_classifier_pl = tf.placeholder(tf.bool, shape=())

            # simple model
            pred, end_points, _ = MODEL.get_model(pointclouds_pl, is_training_classifier_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    # pretrain
    model_path = FLAGS.transformer_model_path
    restore_into_scope(model_path, 'learning_rotation', sess)
    # restore_into_scope(MODEL_PATH, 'dgcnn', sess)
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'rotation_matrics_all': rotation_matrics_all,
           'rotation_plane_all': rotation_plane_all,
           'rotation_matrics_sphere_all': rotation_matrics_sphere_all,
           'rotation_y_all': rotation_y_all,
           'pointclouds_origin_pl': pointclouds_origin_pl,
           'rotation_class_pl': rotation_class_pl,
           'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_transformer_pl':is_training_transformer_pl,
           'is_training_classifier_pl': is_training_classifier_pl,
           'pred_rotation_class': pred_rotation_class,
           'pred': pred,
           'loss': loss,
           }

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training_transformer = False
    is_training_classifier = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(DATA_PATH+TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        # transform_list = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            for vote_idx in range(num_votes):
                rotated_data, _, _ = provider.rotate_point_cloud_by_so3(current_data[start_idx:end_idx, :, :])

                # get rotated data
                feed_dict = {ops['pointclouds_origin_pl']: rotated_data,
                             ops['is_training_transformer_pl']: is_training_transformer}
                pred_rotation_class = sess.run(ops['pred_rotation_class'], feed_dict=feed_dict)
                transformed_points_temp = provider.rotate_data_to_origin_classification_so3(rotated_data,
                                                                                       ops['rotation_matrics_all'],
                                                                                       np.argmax(pred_rotation_class,
                                                                                                 axis=1))

                feed_dict = {ops['pointclouds_pl']: transformed_points_temp,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_classifier_pl']: is_training_classifier}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)

                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # pred_val = np.argmax(batch_pred_classes, 1)
            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END
            
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
                if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                    img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                           SHAPE_NAMES[pred_val[i-start_idx]])
                    img_filename = os.path.join(DUMP_DIR, img_filename)
                    output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                    scipy.misc.imsave(img_filename, output_img)
                    error_cnt += 1
                
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    

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


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=12)
    LOG_FOUT.close()
