import logging
import threading
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
import time
from util.config import load_config
from dataset.factory import create as create_dataset
from nnet.net_factory import pose_net
from nnet.pose_net import get_batch_spec
from util.logging import setup_logging

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr


def setup_preloading(batch_spec):
    for name, spec in batch_spec.items():
        print('batch_name', name, 'batch_spec',spec)
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (
        name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())
    QUEUE_SIZE = 20

    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        print('BATCH_SPECS',names, batch_spec[name])
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    print(placeholders)
    return batch, enqueue_op, placeholders

import cv2
import numpy as np
def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        # for k in food.items():
        #     img = v[0]
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     cv2.imshow('img', v)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()
    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg):
    learning_rate = tf.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op


def train():
    root_name = str(datetime.datetime.now().date()) + '_' + \
        str(datetime.datetime.now().time())
    setup_logging()

    cfg = load_config()

    dataset = create_dataset(cfg)

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)
    losses = pose_net(cfg).train(batch)
    total_loss = losses['total_loss']

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()
    
    tf.compat.v1.enable_resource_variables()
    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    max_iter = int(cfg.multi_step[-1][1])

    display_iters = cfg.display_iters
    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    # Start training
    start = time.time()
    for it in range(max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info("iteration: {} loss: {} lr: {}"
                         .format(it, "{0:.4f}".format(average_loss), current_lr))
        # Save snapshot
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg.snapshot_prefix
            name = model_name + '-' + "0000000{}".format(it)[-7:]
            base = './models/mpii/train/' + name
            os.mkdir(base)
            path_to_save = os.path.join(base, model_name)
            saver.save(sess, path_to_save, global_step=it)
            count = 0
            # print(placeholders)
            inputs_tflite=[]
            tensor_inputs=[]
            for batch, tensor in placeholders.items():
                if count == 0:
                    # tensor = tf.reshape(tensor,shape=[1,200,200,3])
                    print('PRINTING INPUTS',tensor)
                    inputs = tf.compat.v1.saved_model.utils.build_tensor_info(tensor)
                    inputs_tflite.append([inputs])
                    tensor_inputs.append(tensor)
                    # print('PRINTING INPUTS - 2',inputs.tensor_shape)
                    # print('TENSOR',inputs)
                if count == 1:
                    part_score_targets = tf.compat.v1.saved_model.utils.build_tensor_info(
                        tensor)
                    tensor_inputs.append(tensor)
                if count == 2:
                    part_score_weights = tf.compat.v1.saved_model.utils.build_tensor_info(
                        tensor)
                    tensor_inputs.append(tensor)
                    
                if count == 3:
                    locref_targets = tf.compat.v1.saved_model.utils.build_tensor_info(
                        tensor)
                    tensor_inputs.append(tensor)

                if count == 4:
                    locref_mask = tf.compat.v1.saved_model.utils.build_tensor_info(
                        tensor)
                    tensor_inputs.append(tensor)

                count += 1
                # print(tensor_info_input)
                # print(i,j)
            # print('PRINTING PLACEHOLDERS', placeholders)
            # print('PRINTING ONEPL', placeholders['Batch.inputs: 0'])
# TESTING
            tensor_part_pred = tf.get_default_graph().get_tensor_by_name('pose/part_pred/block4/BiasAdd:0')
            part_pred = tf.compat.v1.saved_model.utils.build_tensor_info(tensor_part_pred)
            tensor_locref_pred = tf.get_default_graph().get_tensor_by_name('pose/locref_pred/block4/BiasAdd:0')
            locref_pred = tf.compat.v1.saved_model.utils.build_tensor_info(tensor_locref_pred)
            # print('LOCREF',locref_pred.name)
            
            export_dir = base + '/output'
            builder = tf.compat.v1.saved_model.Builder(export_dir)

            ## Prueba conversor desde sess a tflite
            # outputs=[]
            # outputs.append(tf.identity(locref_pred, name='pose/locref_pred/block4/BiasAdd:0'))  
            # outputs.append(tf.identity(part_pred, name='pose/part_pred/block4/BiasAdd:0'))  
            # inputs={inputs, part_score_targets, part_score_weights,
            #                  locref_targets,locref_mask}
            outputs = []
            outputs.append(tensor_part_pred)
            outputs.append(tensor_locref_pred)

            # outputs={tensor_locref_pred, tensor_part_pred}
            # print(inputs_tflite[0].name)
            # placeholders_list //probable lista de inputs
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, tensor_inputs, outputs)
            converter.allow_custom_ops = True
            # tflite_model = converter.convert()
            # open("converted_model.tflite", "wb").write(tflite_model)
            # with open('model.tflite', 'wb') as f:
            #     f.write(tflite_model)
            
            # prediction_signature = (
            # tf.saved_model.signature_def_utils.build_signature_def(
            #     inputs={'Placeholder:0': inputs, 'Placeholder_1:0': part_score_targets, 'Placeholder_2:0': part_score_weights,
            #                 'Placeholder_3:0': locref_targets, 'Placeholder_4:0': locref_mask},
            #     outputs={'locref_pred': locref_pred, 'part_pred': part_pred},
            #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            # builder.add_meta_graph_and_variables(
            #     sess, [tf.saved_model.tag_constants.SERVING],
            #     signature_def_map={
            #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            #         prediction_signature
            #     },
            # )
# Export checkpoint to SavedModel

            # builder.add_meta_graph_and_variables(
            #     sess, [tf.saved_model.tag_constants.SERVING],
            #     signature_def_map={
            #         'serving_default':
            #         prediction_signature,
            #     })

            # # builder.add_meta_graph(
            #     [tf.saved_model.tag_constants.TRAINING], strip_default_attrs=True)
            # with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            #     builder.add_meta_graph(["bar-tag", "baz-tag"])
            tf.train.write_graph(tf.get_default_graph(), './',
                     'saved_model_m.pb', as_text=False)


            # builder.save()
# /ATTEMPT

            # print('PRUEBA',sess.graph, sess.graph_def)
            # tf.train.write_graph(sess.graph_def, path_to_save,'saved_model.pbtxt', as_text=True)
            # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.graph.as_graph_def(), [final_tensor_name])
            # with gfile.FastGFile('prueba.pb', 'wb') as f:
            #     f.write(output_graph_def.SerializeToString())
            # tS3 = threading.Thread(target=saveToS3, args=(base, name, root_name))
            # tS3.start()

        # if it == 10001:
        #    break
    # Finishing training
    total = time.time() - start
    print("Total training time: {} seconds".format(total))

    sess.close()
    coord.request_stop()
    coord.join([thread])


if __name__ == '__main__':
    start = time.time()
    train()
    total = time.time() - start
    print("Training with initial time: {} seconds".format(total))
