import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
import gzip
from random import random
import sys
from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
#reload(sys)
#sys.setdefaultencoding("utf-8")
import numpy.linalg as linalg
from transformations import euler_from_matrix



class InputHelper(object):

    def setup(self,kitti_odompath, kitti_parentpath, seq_list):
        self.odomDict={}
        self.kitti_odompath = kitti_odompath
        self.kitti_parentpath = kitti_parentpath

        self.setOdomInfo(seq_list)


    def setOdomInfo(self,seq_list):

        for seq in seq_list:
            append_seqname="%02d.txt" % (seq,)
            odom_list=[]
            for line in open(self.kitti_odompath+append_seqname): ##+'/'    ???
                val=line.split()
                val=[float(ele) for ele in val]
                odom_list.append(val)
            self.odomDict[seq]=odom_list

    def getKittiBatch(self,batch_size, sample_range, seq_list, is_train, img_num_dict, conv_model_spec, epoch,  get_img_tforms=1):
        lenseq=len(seq_list)
        seq_idx=np.random.randint(0,lenseq+1)
        seq_num=seq_list[seq_idx]
        labels=[]
        imgpaths_src=[]
        imgpaths_tgt=[]
        seq_path=self.kitti_parentpath+"%02d/" % (seq_num,)
        tforms=[]
        tforms_imgs=[]
        seq_imgs_num=img_num_dict[seq_num]
        odomlist=self.odomDict[seq_num]
        for x in range(batch_size):
            src_img_num=np.random.randint(0,seq_imgs_num) #index check
            radius_num=np.random.randint(1,sample_range+1) #index check for image naming
            odom_src=odomlist[src_img_num]
            odom_src=np.reshape(odom_src,(3,4))
            print(odom_src)
            if random()>0.5:
                if (src_img_num-radius_num)>0:
                    tgt_img_num=src_img_num-radius_num
                else:
                    tgt_img_num=src_img_num+radius_num
            else:
                if (src_img_num+radius_num)<seq_imgs_num-1:
                    tgt_img_num=src_img_num+radius_num
                else:
                    tgt_img_num=src_img_num-radius_num

            odom_tgt=odomlist[tgt_img_num]
            odom_tgt=np.reshape(odom_tgt,(3,4))
            newrow = [0,0,0,1]
            odom_tgt_4x4 = np.vstack([odom_tgt, newrow])
            odom_src_4x4 = np.vstack([odom_src, newrow])
            odom_src_inv=linalg.inv(odom_src_4x4)
            #src inv into tgt - relative transform
            rel_odom_src_tgt=np.matmul(odom_src_inv,odom_tgt_4x4)
            ##convert this to euler
            rel_tform_rot=rel_odom_src_tgt[0:3,0:3]
            rx,ry,rz = euler_from_matrix(rel_tform_rot)
            rel_tform_vec = [ rel_odom_src_tgt[0,4], rel_odom_src_tgt[1,4], rel_odom_src_tgt[2,4], rx, ry, rz]

            rel_tform_vec=rel_odom_src_tgt[0:3,:]
            rel_tform_vec=rel_tform_vec.flatten()
            heightwise = np.tile(rel_tform_vec,(conv_model_spec[1][0],1))
            widthwise = np.tile(heightwise,(1,conv_model_spec[1][1]))
            print(widthwise.shape)
            tforms_imgs.append(widthwise)
            tforms.append(rel_tform_vec)

            src_img_path=seq_path+ 'image_2/' +'%06d.png' % (src_img_num,)
            tgt_img_path=seq_path+ 'image_2/' +'%06d.png' % (tgt_img_num,)
            imgpaths_src.append(src_img_path)
            imgpaths_tgt.append(tgt_img_path)

        imgslist = self.load_preprocess_images_kitti(imgpaths_src,imgpaths_tgt,conv_model_spec,epoch) #return src, tgt

        if get_img_tforms==1:

            return imgslist[0],imgslist[1],tforms_imgs

        else:
            return imgslist[0],imgslist[1],tforms



    def load_preprocess_images_kitti(self, source_paths, target_paths, conv_model_spec, epoch, is_train=True):
        batch1_seq, batch2_seq = [], []
        crop_window=np.random.randint(0,3)

        for src_path,tgt_path in zip(source_paths, target_paths):
            img_org = misc.imread(src_path)
            img_normalized = self.normalize_input(img_org, conv_model_spec,crop_window)
            batch1_seq.append(img_normalized)

            img_org = misc.imread(tgt_path)
            img_normalized = self.normalize_input(img_org, conv_model_spec,crop_window)
            batch2_seq.append(img_normalized)

        #misc.imsave('temp1.png', np.vstack([np.hstack(batch1_seq),np.hstack(batch2_seq)]))

        temp =  [np.asarray(batch1_seq), np.asarray(batch2_seq)]
        return temp


    def batch_iter(self, x1, x2, y, video_lengths, batch_size, num_epochs, conv_model_spec, shuffle=True, is_train=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(y)
        temp = int(data_size/batch_size)
        num_batches_per_epoch = temp+1 if (data_size%batch_size) else temp

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x1_shuffled=x1[shuffle_indices]
                x2_shuffled=x2[shuffle_indices]
                y_shuffled=y[shuffle_indices]
                video_lengths_shuffled = video_lengths[shuffle_indices]
            else:
                x1_shuffled=x1
                x2_shuffled=x2
                y_shuffled=y
                video_lengths_shuffled = video_lengths
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                #print(y_shuffled[start_index:end_index])

                processed_imgs = self.load_preprocess_images(x1_shuffled[start_index:end_index], x2_shuffled[start_index:end_index], conv_model_spec, epoch ,is_train)
                yield( processed_imgs[0], processed_imgs[1]  , y_shuffled[start_index:end_index], video_lengths_shuffled[start_index:end_index])


    def normalize_input(self, img, conv_model_spec,crop_window):
        img = img.astype(dtype=np.float32)
        #img = img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
        vert_offset=100
        horz_offset=0
        img = img[vert_offset:vert_offset+conv_model_spec[1][0],crop_window*conv_model_spec[1][1]:crop_window*conv_model_spec[1][1]+conv_model_spec[1][1]]
        img = img - conv_model_spec[0]

        return img

    # Data Preparatopn
    # ==================================================


    """
    def apply_image_augmentations(self):
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)
        self.train_seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by 0-5% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.21), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                shear=(-10, 10), # shear by -12 to +12 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((2, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.5)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 4.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ],
        random_order=True
        )

    def data_augmentations(self):
        seq_det = []
        for i in range(5):
            seq_det.append(self.train_seq.to_deterministic())
        self.seq_det = seq_det


    """

def save_plot(val1, val2, xlabel, ylabel, title, axis, legend,path):
    pyplot.figure()
    pyplot.plot(val1, '*r--', val2, '^b-')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.axis(axis)
    pyplot.legend(legend)
    pyplot.savefig(path+'.pdf')
    pyplot.clf()

def compute_distance(distance, loss):
    d = np.copy(distance)
    if loss == "AAAI":
        d[distance>=0.5]=1
        d[distance<0.5]=0
    elif loss == "contrastive":
        d[distance>0.5]=0
        d[distance<=0.5]=1
    else:
        raise ValueError("Unkown loss function {%s}".format(loss))
    return d
