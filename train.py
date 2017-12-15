#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
#from helper import InputHelper, save_plot, compute_distance
import gzip
from random import random
from singleview import Net

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 1000, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_file_path", "./", "training folder")
tf.flags.DEFINE_string("kitti_odom_path", "/home/tushar/dataset/Datasets/Kitti_BagFiles/dataset/poses/", "training folder")
tf.flags.DEFINE_string("kitti_parentpath", "/home/tushar/dataset/Datasets/Kitti_BagFiles/dataset/sequences/", "training folder")

tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")
tf.flags.DEFINE_string("name", "result", "prefix names of the output files(default: result)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 100)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss function")
tf.flags.DEFINE_boolean("is_train", False, "Training ConvNet (Default: False)")
tf.flags.DEFINE_float("lr", 0.00001, "learning-rate(default: 0.00001)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("summaries_dir", "./../summaries/", "Summary storage")

#Model Parameters
tf.flags.DEFINE_string("checkpoint_path", "", "pre-trained checkpoint path")
tf.flags.DEFINE_integer("numseqs", 11, "kitti sequences")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_file_path==None:
    print("Input Files List is empty. use --training_file_path argument.")
    exit()


seqs=[i for i in range(0,FLAGS.numseqs) ] ##can choose spcific seqs also instead of this
#break into train and test
seqstrain=seqs[0:10]
seqstest=seqs[10:]

inpH = InputHelper()
inpH.setup(FLAGS.kitti_odom_path, FLAGS.kitti_parentpath ,seqs)
train_set, dev_set, sum_no_of_batches = inpH.getDataSets(FLAGS.training_file_path, FLAGS.max_frames, 50, FLAGS.batch_size)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options,
      )
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        convModel = Conv(
         FLAGS.conv_layer,
         FLAGS.conv_layer_weight_pretrained_path,
         FLAGS.batch_size,
         FLAGS.max_frames,
         FLAGS.conv_net_training)

        siameseModel = SiameseLSTM(
            sequence_length=FLAGS.max_frames,
            input_size=9216,
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            num_lstm_layers=FLAGS.num_lstm_layers,
            hidden_unit_dim=FLAGS.hidden_dim,
            loss=FLAGS.loss,
            projection=FLAGS.projection,
            return_outputs=FLAGS.return_outputs)
        

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate=tf.train.exponential_decay(1e-5, global_step, sum_no_of_batches*5, 0.95, staircase=False, name=None)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        print("initialized convmodel and siamesemodel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    #grad_summaries_merged = tf.summary.merge(grad_summaries)
    summaries_merged = tf.summary.merge_all()
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/data4/abhijeet/gta/", "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    #lstm_checkpoint_prefix = os.path.join(checkpoint_dir, "lstm_model")
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    lstm_savepath="/data4/abhijeet/gta/lstm_outputs"
    if not os.path.exists(lstm_savepath):
            os.makedirs(lstm_savepath)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    #lstm_saver = tf.train.Saver([out1,out2], max_to_keep=2)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    #Fix weights for Conv Layers
    convModel.initalize(sess)

    #print all trainable parameters
    tvar = tf.trainable_variables()
    for i, var in enumerate(tvar):
        print("{}".format(var.name))
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val' , graph=tf.get_default_graph())
    
    def train_step(x1_batch, x2_batch, y_batch, video_lengths):
        
        #A single training step
        
        [x1_batch] = sess.run([convModel.features],  feed_dict={convModel.input_imgs: x1_batch})
        [x2_batch] = sess.run([convModel.features],  feed_dict={convModel.input_imgs: x2_batch})

        feed_dict = {
                         siameseModel.input_x1: x1_batch,
                         siameseModel.input_x2: x2_batch,
                         siameseModel.input_y: y_batch,
                         siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel.video_lengths: video_lengths,
        }

        out1, out2, _, step, loss, dist, summary = sess.run([siameseModel.out1, siameseModel.out2, tr_op_set, global_step, siameseModel.loss, siameseModel.distance, summaries_merged],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print(out1, out2)
        #print(video_lengths)
        #print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        print(y_batch, dist, d)
        return summary, np.sum(correct), loss

    def dev_step(x1_batch, x2_batch, y_batch, video_lengths, dev_iter, epoch):
        
        #A single training step
        
        [x1_batch] = sess.run([convModel.features],  feed_dict={convModel.input_imgs: x1_batch})
        [x2_batch] = sess.run([convModel.features],  feed_dict={convModel.input_imgs: x2_batch})

        feed_dict = {
                         siameseModel.input_x1: x1_batch,
                         siameseModel.input_x2: x2_batch,
                         siameseModel.input_y: y_batch,
                         siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel.video_lengths: video_lengths,

        }
       
        step, loss, dist, summary, out1, out2 = sess.run([global_step, siameseModel.loss, siameseModel.distance, summaries_merged,siameseModel.out1,siameseModel.out2],  feed_dict)
        #np.save(lstm_savepath+'/out1_'+str(dev_iter)+'_'+str(epoch),out1)
        #np.save(lstm_savepath+'/out2_'+str(dev_iter)+'_'+str(epoch),out2)
        #np.save(lstm_savepath+'/y_'+str(dev_iter)+'_'+str(epoch),y_batch)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        #print(y_batch, dist, d)
        return summary, np.sum(correct), loss, correct

    # Generate batches
    batches=inpH.batch_iter(
                train_set[0], train_set[1], train_set[2], train_set[3], FLAGS.batch_size, FLAGS.num_epochs, convModel.spec, shuffle=True, is_train=False)

    ptr=0
    max_validation_correct=0.0
    start_time = time.time()
    train_accuracy, val_accuracy = [] , []
    train_loss, val_loss = [], []
    train_batch_loss_arr, val_batch_loss_arr = [], []


    for nn in xrange(FLAGS.num_epochs):
        # Evaluate on Validataion Data for every epoch
        sum_val_correct=0.0
        val_epoch_loss=0.0
        val_results = []
        print("\nEvaluation:")
        dev_batches = inpH.batch_iter(dev_set[0],dev_set[1],dev_set[2],dev_set[3], FLAGS.batch_size, 1, convModel.spec, shuffle=False , is_train=False)
        dev_iter=0
        for (x1_dev_b,x2_dev_b,y_dev_b, dev_video_lengths) in dev_batches:
            if len(y_dev_b)<1:
                continue
            dev_iter += 1
            summary, batch_val_correct , val_batch_loss, batch_results = dev_step(x1_dev_b, x2_dev_b, y_dev_b, dev_video_lengths, dev_iter,nn)
            val_results = np.concatenate([val_results, batch_results])
            sum_val_correct = sum_val_correct + batch_val_correct
            current_step = tf.train.global_step(sess, global_step)
            val_writer.add_summary(summary, current_step)
            val_epoch_loss = val_epoch_loss + val_batch_loss*len(y_dev_b)
            val_batch_loss_arr.append(val_batch_loss*len(y_dev_b))
        print("val_loss ={}".format(val_epoch_loss/len(dev_set[2])))
        print("total_val_correct={}/total_val={}".format(sum_val_correct, len(dev_set[2])))
        val_accuracy.append(sum_val_correct*1.0/len(dev_set[2]))
        val_loss.append(val_epoch_loss/len(dev_set[2]))
    
        print("Epoch Number: {}".format(nn))
        epoch_start_time = time.time()
        sum_train_correct=0.0
        train_epoch_loss=0.0
        for kk in xrange(sum_no_of_batches):
            x1_batch, x2_batch, y_batch, video_lengths = batches.next()
            if len(y_batch)<1:
                continue
            summary, train_batch_correct, train_batch_loss =train_step(x1_batch, x2_batch, y_batch, video_lengths)
            train_writer.add_summary(summary, current_step)
            sum_train_correct = sum_train_correct + train_batch_correct    
            train_epoch_loss = train_epoch_loss + train_batch_loss* len(y_batch)
            train_batch_loss_arr.append(train_batch_loss*len(y_batch))
        print("train_loss ={}".format(train_epoch_loss/len(train_set[2])))
        print("total_train_correct={}/total_train={}".format(sum_train_correct, len(train_set[2])))
        train_accuracy.append(sum_train_correct*1.0/len(train_set[2]))
        train_loss.append(train_epoch_loss/len(train_set[2]))

        # Update stored model
        if current_step % (FLAGS.checkpoint_every) == 0:
            if sum_val_correct >= max_validation_correct:
                max_validation_correct = sum_val_correct
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                #lstm_saver.save(sess, lstm_checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with checkpoint to {}".format(nn, checkpoint_prefix))

        epoch_end_time = time.time()
        print("Total time for {} th-epoch is {}\n".format(nn, epoch_end_time-epoch_start_time))
        save_plot(train_accuracy, val_accuracy, 'epochs', 'accuracy', 'Accuracy vs epochs', [-0.1, nn+0.1, 0, 1.01],  ['train','val' ],'./accuracy_'+str(FLAGS.name))
        save_plot(train_loss, val_loss, 'epochs', 'loss', 'Loss vs epochs', [-0.1, nn+0.1, 0, np.max(train_loss)+0.2],  ['train','val' ],'./loss_'+str(FLAGS.name))
        save_plot(train_batch_loss_arr, val_batch_loss_arr, 'steps', 'loss', 'Loss vs steps', [-0.1, (nn+1)*sum_no_of_batches+0.1, 0, np.max(train_batch_loss_arr)+0.2],  ['train','val' ],'./loss_batch_'+str(FLAGS.name))

    end_time = time.time()
    print("Total time for {} epochs is {}".format(FLAGS.num_epochs, end_time-start_time))

#"""
