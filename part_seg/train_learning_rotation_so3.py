import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import pdb
import tf_util
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data_path', default='/mnt/dengshuang/data/shapenet50/shapenet_hdf5_data', help='data path')
parser.add_argument('--model', default='learning_rotation_nets_mix_model', help='Model name: dgcnn')
parser.add_argument('--log_dir', default='result/log_learning_rotation_so3', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_learning_rotation_so3.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 10

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ShapeNet50 official train/test split
hdf5_data_dir = FLAGS.data_path
all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()
all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)
TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')
train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)
train_file_idx = np.arange(0, len(train_file_list))
data = []
label = []
for i in range(num_train_file):
    cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])
    print('Loading train file ' + cur_train_filename)
    cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_train_filename)
    data.append(cur_data)
    label.append(cur_labels)

test_file_idx = np.arange(0, len(test_file_list))
test_data = []
test_label = []
for i in range(num_test_file):
    cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[test_file_idx[i]])
    print('Loading train file ' + cur_test_filename)
    cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_test_filename)
    test_data.append(cur_data)
    test_label.append(cur_labels)



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            rotation_class_num, rotation_matrics_all, rotation_plane_all, rotation_matrics_sphere_all, rotation_y_all = provider.get_so3_rotation_matrics_all()
            pointclouds_pl, rotation_class_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred_rotation_class, _ = MODEL.get_model(pointclouds_pl, rotation_class_num, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred_rotation_class, rotation_class_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred_rotation_class, 1), tf.to_int64(rotation_class_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)

        sess.run(init, {is_training_pl: True})

        ops = {'data': data,
               'test_data': test_data,
               'rotation_matrics_all': rotation_matrics_all,
               'rotation_plane_all': rotation_plane_all,
               'rotation_matrics_sphere_all': rotation_matrics_sphere_all,
               'rotation_y_all': rotation_y_all,
               'pointclouds_pl': pointclouds_pl,
               'rotation_class_pl': rotation_class_pl,
               'pred_rotation_class': pred_rotation_class,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        begin_epoch = 0
        min_eval_loss = 1e6
        min_eval_epoch = -1
        for epoch in range(begin_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_loss = eval_one_epoch(sess, ops, test_writer)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                min_eval_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))
                log_string("Best Model saved in file: %s" % save_path)
            log_string('min eval loss: %f' % (min_eval_loss))
            log_string('the epoch of min eval loss: %d' % (min_eval_epoch))

            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, num_train_file)
    np.random.shuffle(train_file_idxs)
    
    for fn in range(num_train_file):
        log_string('----' + str(fn) + '-----')

        current_data = ops['data'][fn]

        current_data, current_rotation_matrics, _ = provider.rotate_point_cloud_by_so3(current_data)
        current_rotation_class = provider.get_so3_rotation_class(current_rotation_matrics,
                                                                 ops['rotation_matrics_all'],
                                                                 ops['rotation_plane_all'],
                                                                 ops['rotation_matrics_sphere_all'],
                                                                 ops['rotation_y_all'])

        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_rotation_class, _ = \
            provider.shuffle_data_2(current_data, current_rotation_class)

        file_size = current_data.shape[0]
        print(file_size)
        num_batches = file_size // BATCH_SIZE

        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = current_data[start_idx:end_idx, :, :]
            rotation_class = current_rotation_class[start_idx:end_idx]

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['rotation_class_pl']: rotation_class,
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0
    total_seen = 0
    
    for fn in range(num_test_file):
        log_string('----' + str(fn) + '-----')

        current_data = ops['test_data'][fn]

        current_data, current_rotation_matrics, _ = provider.rotate_point_cloud_by_so3(current_data)
        current_rotation_class = provider.get_so3_rotation_class(current_rotation_matrics,
                                                                 ops['rotation_matrics_all'],
                                                                 ops['rotation_plane_all'],
                                                                 ops['rotation_matrics_sphere_all'],
                                                                 ops['rotation_y_all'])

        current_data = current_data[:, 0:NUM_POINT, :]
        
        file_size = current_data.shape[0]
        print(file_size)
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            rotated_data = current_data[start_idx:end_idx, :, :]
            rotation_class = current_rotation_class[start_idx:end_idx]

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['rotation_class_pl']: rotation_class,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_rotation_class = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred_rotation_class']], feed_dict=feed_dict)

            transformed_points = provider.rotate_data_to_origin_classification_so3(rotated_data, ops['rotation_matrics_all'], np.argmax(pred_rotation_class, axis=1))
            if fn == 0 and batch_idx == 0:
                print(current_rotation_class[:32])
                print(np.argmax(pred_rotation_class[:32], axis=1))
                # pdb.set_trace()

            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    return loss_sum / float(total_seen)


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
