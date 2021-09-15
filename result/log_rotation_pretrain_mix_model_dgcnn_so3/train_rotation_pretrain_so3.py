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
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pdb
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--transformer_model', default='learning_rotation_nets_mix_model', help='Transformer model name: dgcnn [default: dgcnn]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
parser.add_argument('--transformer_model_path', default='result/log_learning_rotation_mix_model_so3_30_30_1000/best_model.ckpt', help='transformer model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='result/log_rotation_pretrain_mix_model_dgcnn_so3_30_30_1000_nonoise', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
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
TRANSFORMER_MODEL = importlib.import_module(FLAGS.transformer_model)
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
TRANSFORMER_MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.transformer_model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (TRANSFORMER_MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_rotation_pretrain_so3.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, '/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '/mnt/dengshuang/data/modelnet40/modelnet40_ply_hdf5_2048/test_files.txt'))


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


def create_output(points1):
    point_num = points1.shape[0]
    colors1 = np.tile(np.array([0, 1, 0], np.int) * 255, [point_num, 1])

    points = points1
    colors = colors1

    vertices = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])

    filename = 'temp.ply'
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
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
                                    		''' % (point_num)
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


TRANSFORM_PLY_DIR = os.path.join(LOG_DIR, 'transform_ply')
ply_dict2 = {}
ply_dict2['save_ply_dir'] = TRANSFORM_PLY_DIR
# if not os.path.exists(ply_dict2['save_ply_dir']): os.mkdir(ply_dict2['save_ply_dir'])
ply_dict2['row_index'] = 0
ply_dict2['col_index'] = 0
ply_dict2['index'] = 0
ply_dict2['gap'] = 2
ply_dict2['max_index'] = 10
ply_dict2['vertices_list'] = []
ply_dict2['ply_index'] = 0
def create_output_transform(points1, points2):
    shape_num = points1.shape[0]
    for i in range(shape_num):
        point_num = points1.shape[1]
        colors1 = np.tile(np.array([1, 0, 0], np.int)*255, [point_num, 1])
        colors2 = np.tile(np.array([0, 1, 0], np.int)*255, [point_num, 1])

        points = np.vstack([points1[i]+np.array([ply_dict2['row_index']*ply_dict2['gap'], ply_dict2['col_index']*ply_dict2['gap'], 0]),
                            points2[i]+np.array([ply_dict2['row_index']*ply_dict2['gap'], ply_dict2['col_index']*ply_dict2['gap'], 0])])
        # points = np.vstack([points1[i],
        #                     points2[i]])
        colors = np.vstack([colors1, colors2])

        vertices = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])

        if ply_dict2['col_index'] < ply_dict2['max_index'] -1:
            ply_dict2['vertices_list'].append(vertices)
            ply_dict2['col_index'] = ply_dict2['col_index'] + 1
            ply_dict2['index'] += 1
            if i == shape_num - 1:
                vertices_all = np.vstack(ply_dict2['vertices_list'])
                if not os.path.exists(ply_dict2['save_ply_dir']): os.mkdir(ply_dict2['save_ply_dir'])
                filename = ply_dict2['save_ply_dir'] + '/' + str(ply_dict2['ply_index']) + '.ply'
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
                                		''' % (ply_dict2['index'] * NUM_POINT * 2)
                with open(filename, 'r+') as f:
                    old = f.read()
                    f.seek(0)
                    f.write(ply_header % dict(vert_num=len(vertices)))
                    f.write(old)
                ply_dict2['vertices_list'] = []
                ply_dict2['col_index'] = 0
                ply_dict2['row_index'] = 0
                ply_dict2['index'] = 0
                ply_dict2['ply_index'] = ply_dict2['ply_index'] + 1
        elif ply_dict2['row_index'] < ply_dict2['max_index'] -1:
            ply_dict2['vertices_list'].append(vertices)
            ply_dict2['col_index'] = 0
            ply_dict2['row_index'] = ply_dict2['row_index'] + 1
            ply_dict2['index'] += 1
            if i == shape_num - 1:
                vertices_all = np.vstack(ply_dict2['vertices_list'])
                if not os.path.exists(ply_dict2['save_ply_dir']): os.mkdir(ply_dict2['save_ply_dir'])
                filename = ply_dict2['save_ply_dir'] + '/' + str(ply_dict2['ply_index']) + '.ply'
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
                        ply_dict2['index'] * NUM_POINT * 2)
                with open(filename, 'r+') as f:
                    old = f.read()
                    f.seek(0)
                    f.write(ply_header % dict(vert_num=len(vertices)))
                    f.write(old)
                ply_dict2['vertices_list'] = []
                ply_dict2['col_index'] = 0
                ply_dict2['row_index'] = 0
                ply_dict2['index'] = 0
                ply_dict2['ply_index'] = ply_dict2['ply_index'] + 1
        else:
            ply_dict2['vertices_list'].append(vertices)
            ply_dict2['index'] += 1
            vertices_all = np.vstack(ply_dict2['vertices_list'])
            if not os.path.exists(ply_dict2['save_ply_dir']): os.mkdir(ply_dict2['save_ply_dir'])
            filename = ply_dict2['save_ply_dir'] + '/' + str(ply_dict2['ply_index']) + '.ply'
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
                		''' % (ply_dict2['index']*NUM_POINT*2)
            with open(filename, 'r+') as f:
                old = f.read()
                f.seek(0)
                f.write(ply_header % dict(vert_num=len(vertices)))
                f.write(old)
            ply_dict2['vertices_list'] = []
            ply_dict2['col_index'] = 0
            ply_dict2['row_index'] = 0
            ply_dict2['index'] = 0
            ply_dict2['ply_index'] = ply_dict2['ply_index']+1


def train():
    with tf.Graph().as_default():
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

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred, end_points, _ = MODEL.get_model(pointclouds_pl, is_training_classifier_pl, bn_decay=bn_decay)
                loss = MODEL.get_loss(pred, labels_pl, end_points)
                tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
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

        sess.run(init, {is_training_transformer_pl: False, is_training_classifier_pl: True})

        # get class data
        data, label = provider.loadDataFile_list_all(TRAIN_FILES)
        test_data, test_label = provider.loadDataFile_list_all(TEST_FILES)

        ops = {'data': data,
               'label': label,
               'test_data': test_data,
               'test_label': test_label,
               'rotation_matrics_all': rotation_matrics_all,
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
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # pretrain
        model_path = FLAGS.transformer_model_path
        restore_into_scope(model_path, 'learning_rotation', sess)

        begin_epoch = 0
        max_eval_acc = -1
        max_eval_epoch = -1
        for epoch in range(begin_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_acc = eval_one_epoch(sess, ops, test_writer)
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                max_eval_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))
                log_string("Best Model saved in file: %s" % save_path)
            log_string('max eval accuracy: %f' % (max_eval_acc))
            log_string('the epoch of max eval accuracy: %d' % (max_eval_epoch))

            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training_transformer = False
    is_training_classifier = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        # current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = ops['data'][fn]
        current_label = ops['label'][fn]

        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        print(file_size)
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data, _, _ = provider.rotate_point_cloud_by_so3(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)

            # get rotated data
            feed_dict = {ops['pointclouds_origin_pl']: jittered_data,
                         ops['is_training_transformer_pl']: is_training_transformer}
            pred_rotation_class = sess.run(ops['pred_rotation_class'], feed_dict=feed_dict)
            # transformed_points = provider.rotate_data_to_origin_classification_z(jittered_data,
            #                                                                      ops['rotation_matrics_all'],
            #                                                                      np.argmax(pred_rotation_class, axis=1))
            transformed_points = provider.rotate_data_to_origin_classification_so3(jittered_data,
                                                                                   ops['rotation_matrics_all'],
                                                                                   np.argmax(pred_rotation_class,
                                                                                             axis=1))

            # create_output_transform(jittered_data, transformed_points)
            # pdb.set_trace()

            # transformed_points, _, _ = provider.rotate_point_cloud_by_so3_small_angle(transformed_points)

            feed_dict = {ops['pointclouds_pl']: transformed_points,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_classifier_pl']: is_training_classifier,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training_transformer = False
    is_training_classifier = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        # current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = ops['test_data'][fn]
        current_label = ops['test_label'][fn]

        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            rotated_data, _, _ = provider.rotate_point_cloud_by_so3(current_data[start_idx:end_idx, :, :])

            # get rotated data
            feed_dict = {ops['pointclouds_origin_pl']: rotated_data,
                         ops['is_training_transformer_pl']: is_training_transformer}
            pred_rotation_class = sess.run(ops['pred_rotation_class'], feed_dict=feed_dict)
            # transformed_points = provider.rotate_data_to_origin_classification_z(rotated_data,
            #                                                                      ops['rotation_matrics_all'],
            #                                                                      np.argmax(pred_rotation_class, axis=1))
            transformed_points = provider.rotate_data_to_origin_classification_so3(rotated_data,
                                                                                   ops['rotation_matrics_all'],
                                                                                   np.argmax(pred_rotation_class,
                                                                                             axis=1))

            create_output_transform(rotated_data, transformed_points)
            # pdb.set_trace()

            feed_dict = {ops['pointclouds_pl']: transformed_points,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_classifier_pl']: is_training_classifier,}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)

            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    ply_dict2['row_index'] = 0
    ply_dict2['col_index'] = 0
    ply_dict2['vertices_list'] = []
    ply_dict2['ply_index'] = 0
    # pdb.set_trace()
    return total_correct / float(total_seen)


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


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
