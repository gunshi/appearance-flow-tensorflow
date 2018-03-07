#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from helper_tvsn import InputHelper, save_plot
import gzip
from random import random
from tvsn_ae import Net_tvsn
from scipy.misc import imsave
import sys
# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 1000, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("exp_reg_weight", 0.0, "L2 regularizaion lambda (default: 0.0)")

tf.flags.DEFINE_string("dataset_to_use", "SYNTHIA", "training folder")

#kitti paths
tf.flags.DEFINE_string("kitti_odom_path", "/home/tushar/dataset/Datasets/Kitti_BagFiles/dataset/poses/", "training folder")
tf.flags.DEFINE_string("kitti_parentpath", "/home/tushar/dataset/Datasets/Kitti_BagFiles/dataset/sequences/", "training folder")

#synthia paths
tf.flags.DEFINE_string("synthia_parentpath", "/scratch/tushar.vaidya/synthia/", "training folder")
tf.flags.DEFINE_string("synthia_configpath", "SYNTHIA_data.txt", "training folder")
tf.flags.DEFINE_string("synthia_frame_info_path", "FRAMES_SYNTHIA.txt", "training folder")
tf.flags.DEFINE_string("synthia_odom_path", "CameraParams/Stereo_Left/", "training folder")
tf.flags.DEFINE_string("synthia_rgb_path", "RGB/Stereo_Left/", "training folder")
tf.flags.DEFINE_string("synthia_semseg_path", "GT/COLOR/Stereo_Left/", "training folder")
tf.flags.DEFINE_string("synthia_depth_path", "Depth/Stereo_Left/", "training folder")
tf.flags.DEFINE_string("synthia_output_save_path", "/scratch/tushar.vaidya/afn/outputs/", "training folder")
tf.flags.DEFINE_string("synthia_image_save_path", "/scratch/tushar.vaidya/afn/outputs/imgs/", "training folder")


tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")
tf.flags.DEFINE_string("name", "result", "prefix names of the output files(default: result)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("sample_range", 10, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 100)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss function")
tf.flags.DEFINE_boolean("is_train", False, "Training ConvNet (Default: False)")
tf.flags.DEFINE_float("lr", 0.0001, "learning-rate(default: 0.00001)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("summaries_dir", "/scratch/tushar.vaidya/afn/outputs/summaries/", "Summary storage")
tf.flags.DEFINE_string("outputs_dir", "/scratch/tushar.vaidya/afn/outputs/", "Summary storage")

#Model Parameters
tf.flags.DEFINE_string("checkpoint_path", "./", "pre-trained checkpoint path")
tf.flags.DEFINE_integer("numseqs", 11, "kitti sequences")
tf.flags.DEFINE_integer("batches_train", 900 , "batches for train")
tf.flags.DEFINE_integer("batches_test", 50, "batches for test")
tf.flags.DEFINE_boolean("conv_net_training", True, "Training ConvNet (Default: False)")
tf.flags.DEFINE_boolean("multi_view_training", False, "Training ConvNet (Default: False)")

FLAGS = tf.flags.FLAGS
#FLAGS(sys.argv)
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.kitti_odom_path==None:
    print("Input Files List is empty. use --training_file_path argument.")
    exit()

if FLAGS.kitti_parentpath==None:
    print("Input Files List is empty. use --training_file_path argument.")
    exit()



inpH = InputHelper()


if(FLAGS.dataset_to_use=='KITTI'):
    seqs=[ i for i in range(0,FLAGS.numseqs) ]
    seqstrain = seqs[0:10]
    seqstest = seqs[10:]
    imgs_counts={0:4540,1:1100,2:4660,3:800,4:270,5:2760,6:1100,7:1100,8:4070,9:1590,10:1200,11:920}
    inpH.setup(FLAGS.kitti_odom_path, FLAGS.kitti_parentpath ,seqs)

if(FLAGS.dataset_to_use=='SYNTHIA'):
    seqstest = [4]
    seqstrain = [1,2]
    imgs_counts = {1:{'DAWN':1451,'NIGHT':935,'SUMMER':943,'SPRING':1189},2:{'SUMMER':888,'FALL':742,'SPRING':969,'NIGHT':720},4:{'SUMMER':901, 'SPRING':959,'SUNSET':958 },5:{'SPRING':295,'SUNSET':707,'SUMMER':787},6:{'SUNSET':841,'SUMMER':1014,'SPRING':1044,'NIGHT':850}}
#DAWN 4 850
    inpH.setup_synthia_sparse(FLAGS.synthia_odom_path, FLAGS.synthia_parentpath, FLAGS.synthia_rgb_path, imgs_counts)


# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options,
      )
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        if(FLAGS.multi_view_training):
            convModel = Net_tvsn(
                 FLAGS.batch_size,
                 FLAGS.conv_net_training,
                 FLAGS.exp_reg_weight)
        else:

            convModel = Net_tvsn(
             FLAGS.batch_size,
             FLAGS.conv_net_training,
             FLAGS.exp_reg_weight)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #learning_rate=tf.train.exponential_decay(1e-5, global_step, sum_no_of_batches*5, 0.95, staircase=False, name=None)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        print("initialized Net object")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
    	grads_and_vars=optimizer.compute_gradients(convModel.loss)
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
    summaries_merged = tf.summary.merge_all()
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.outputs_dir, "runs", FLAGS.name))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    current_path = os.getcwd()
 
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    #Fix weights for Conv Layers
    #convModel.initalize(sess)

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

    def train_step(src_batch, tgt_batch, tform_batch, train_iter, epoch, multi_view_training):

        #A single training step
        if(FLAGS.multi_view_training):

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.aux_imgs: src_batch[1],
                        convModel.tgt_imgs: tgt_batch,
                        convModel.tform: tform_batch[0],
                        convModel.tform_aux: tform_batch[1], 
                    	convModel.keep_prob: FLAGS.dropout_keep_prob,
                        convModel.is_train:True,
			convModel.phase:True}


        else:

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.tgt_imgs: tgt_batch,
                        convModel.tform: tform_batch[0],
                    	convModel.keep_prob: FLAGS.dropout_keep_prob,
                        convModel.is_train:True,
			convModel.phase:True}



        if(train_iter%800==0):
            outputs,masks, _, step, loss, summary = sess.run([convModel.tgts,convModel.masks, tr_op_set, global_step, convModel.loss, summaries_merged],  feed_dict)
            img_num=0
	    #process outputs to get them in range, swap channels

            for i in range(len(outputs)):
		#outputs[i][:,:,:] = outputs[i][:,:,::-1]
		#outputs[i] = (outputs[i]*127.5)+127.5
		#masks[i] = masks[i]*255
                imsave(FLAGS.synthia_image_save_path+str(train_iter)+'_'+str(img_num)+'_mask.png', masks[i])

                imsave(FLAGS.synthia_image_save_path+str(train_iter)+'_'+str(img_num)+'_output.png', outputs[i])
                imsave(FLAGS.synthia_image_save_path+str(train_iter)+'_'+str(img_num)+'_target.png', tgt_batch[i])
                for j in range(len(src_batch)):
                    imsave(FLAGS.synthia_image_save_path+str(train_iter)+'_'+str(img_num)+'_input'+str(j)+'.png', src_batch[j][i])
                img_num+=1
        else:
             _, step, loss, summary = sess.run([tr_op_set, global_step, convModel.loss, summaries_merged],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        return summary, loss

    def dev_step(src_batch, tgt_batch, tform_batch, dev_iter, epoch, multi_view_training):

        #A single validation step

        if(FLAGS.multi_view_training):

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.aux_imgs: src_batch[1],
                        convModel.tgt_imgs: tgt_batch,
                        convModel.tform: tform_batch[0],
                        convModel.tform_aux: tform_batch[1] ,
                        convModel.keep_prob: FLAGS.dropout_keep_prob,
                        convModel.is_train:False,
			convModel.phase:False
                        }

        else:

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.tgt_imgs: tgt_batch,
                        convModel.tform: tform_batch[0], 
                        convModel.keep_prob: FLAGS.dropout_keep_prob,
                        convModel.is_train:False,
			convModel.phase:False
                        }



        outputs,masks,step, loss, summary, outputs= sess.run([convModel.tgts,convModel.masks,global_step, convModel.loss, summaries_merged,convModel.tgts],  feed_dict)
        img_num=0
        for i in range(len(outputs)):
            #outputs[i][:,:,:] = outputs[i][:,:,::-1]
            #outputs[i] = (outputs[i]*127.5)+127.5
            #masks[i] = masks[i]*255
            imsave(FLAGS.synthia_image_save_path+str(dev_iter)+'_'+str(img_num)+'_maskeval.png', masks[i])

            imsave(FLAGS.synthia_image_save_path+str(dev_iter)+'_'+str(img_num)+'_outputeval.png', outputs[i])
            imsave(FLAGS.synthia_image_save_path+str(dev_iter)+'_'+str(img_num)+'_targeteval.png', tgt_batch[i])
            for j in range(len(src_batch)):
                imsave(FLAGS.synthia_image_save_path+str(dev_iter)+'_'+str(img_num)+'_inputeval'+str(j)+'.png', src_batch[j][i])
            img_num+=1

        time_str = datetime.datetime.now().isoformat()

        return summary, loss

    def get_batch_appropriate_train(seqstrain, is_train, imgs_counts, spec, nn, multi_view_training):
        if(FLAGS.dataset_to_use=='SYNTHIA'):
            return inpH.getSynthiaBatch(FLAGS.batch_size,FLAGS.sample_range,seqstrain,is_train, spec,nn, multi_view_training)
        else:
            return inpH.getKittiBatch(FLAGS.batch_size,FLAGS.sample_range,seqstrain,is_train, imgs_counts, spec,nn, FLAGS.multi_view_training)
    
    def get_batch_appropriate_test(seqstest, is_train, imgs_counts, spec, nn, multi_view_training):
        if(FLAGS.dataset_to_use=='SYNTHIA'):
            return inpH.getSynthiaBatch(FLAGS.batch_size,FLAGS.sample_range,seqstest,is_train, spec, nn, multi_view_training)
        else:
            return inpH.getKittiBatch(FLAGS.batch_size,FLAGS.sample_range,seqstest,is_train, imgs_counts, spec, nn, FLAGS.multi_view_training)

    start_time = time.time()
    train_loss, val_loss = [], []
    train_batch_loss_arr, val_batch_loss_arr = [], []


    for nn in range(FLAGS.num_epochs):

        current_step = tf.train.global_step(sess, global_step)
        print("Epoch Number: {}".format(nn))
        epoch_start_time = time.time()
        train_epoch_loss=0.0
        for kk in range(FLAGS.batches_train):
            print(str(kk))
            src_batch, tgt_batch, tform_batch = get_batch_appropriate_train(seqstrain,True, imgs_counts, convModel.spec,nn, FLAGS.multi_view_training)
            if len(tform_batch)<1:
                continue
            summary, train_batch_loss =train_step(src_batch, tgt_batch, tform_batch, kk, nn, FLAGS.multi_view_training)
            train_writer.add_summary(summary, current_step)
            train_epoch_loss = train_epoch_loss + train_batch_loss* len(tform_batch)
            train_batch_loss_arr.append(train_batch_loss*len(tform_batch))
        print("train_loss ={}".format(train_epoch_loss/(FLAGS.batches_train*FLAGS.batch_size)))
        train_loss.append(train_epoch_loss/(FLAGS.batches_train*FLAGS.batch_size))

        # Evaluate on Validataion Data for every epoch
        val_epoch_loss=0.0
        print("\nEvaluation:")

        for kk in range(FLAGS.batches_test):
            src_dev_b, tgt_dev_b, tform_dev_b = get_batch_appropriate_test(seqstest,True, imgs_counts, convModel.spec, nn, FLAGS.multi_view_training)

            summary,  val_batch_loss = dev_step(src_dev_b, tgt_dev_b, tform_dev_b, kk ,nn, FLAGS.multi_view_training)

            val_writer.add_summary(summary, current_step)
            val_epoch_loss = val_epoch_loss + val_batch_loss*len(tform_dev_b)
            val_batch_loss_arr.append(val_batch_loss*len(tform_dev_b))
            print("val_loss ={}".format(val_epoch_loss/FLAGS.batch_size*FLAGS.batches_test))
        val_loss.append(val_epoch_loss/FLAGS.batch_size*FLAGS.batches_test)



        # Update stored model
        if current_step % (FLAGS.checkpoint_every) == 0:
            saver.save(sess, checkpoint_prefix, global_step=current_step)
            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
            print("Saved model {} with checkpoint to {}".format(nn, checkpoint_prefix))

        epoch_end_time = time.time()
        empty=[]
        print("Total time for {} th-epoch is {}\n".format(nn, epoch_end_time-epoch_start_time))
        save_plot(train_loss, val_loss, 'epochs', 'loss', 'Loss vs epochs', [-0.1, nn+0.1, 0, np.max(train_loss)+0.2],  ['train','val' ],'./loss_'+str(FLAGS.name))
        save_plot(train_batch_loss_arr, val_batch_loss, 'steps', 'loss', 'Loss vs steps', [-0.1, (nn+1)*FLAGS.batch_size*FLAGS.batches_test+0.1, 0, np.max(train_batch_loss_arr)+0.2],  ['train','val' ],'./loss_batch_'+str(FLAGS.name))

    end_time = time.time()
    print("Total time for {} epochs is {}".format(FLAGS.num_epochs, end_time-start_time))

