import argparse
from ..config import default, generate_config
from ..symbol import *
from ..utils.load_model import load_param
from ..core.module import MutableModule
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper
from minimum_bounding import minimum_bounding_rectangle
bbox_pred = nonlinear_pred

import numpy as np
import os
from scipy import io
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile

def demo_maskrcnn(network, ctx, prefix, epoch,
                   vis= True, has_rpn = True, thresh = 0.001):
    
    assert has_rpn,"Only has_rpn==True has been supported."
    sym = eval('get_' + network + '_mask_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
    split = False
    max_image_shape = (1,3,1024,1024)
    max_data_shapes = [("data",max_image_shape),("im_info",(1,3))]
    mod = MutableModule(symbol = sym, data_names = ["data","im_info"], label_names= None,
                            max_data_shapes = max_data_shapes,
                              context=ctx)
    mod.bind(data_shapes = max_data_shapes, label_shapes = None, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    class OneDataBatch():
        def __init__(self,img):
            im_info = mx.nd.array([[img.shape[0],img.shape[1],1.0]])
            img = np.transpose(img,(2,0,1)) 
            img = img[np.newaxis,(2,1,0)]
            self.data = [mx.nd.array(img),im_info]
            self.label = None
            self.provide_label = None
            self.provide_data = [("data",(1,3,img.shape[2],img.shape[3])),("im_info",(1,3))]

    imglist_file = os.path.join(default.dataset_path, 'imglists', 'test.lst')
    #print(default.dataset_path)
    assert os.path.exists(imglist_file), 'Path does not exist: {}'.format(imglist_file)
    imgfiles_list = []
    with open(imglist_file, 'r') as f:
        for line in f:
            file_list = dict()
            label = line.strip().split('\t')
            file_list['img_path'] = label[1]
            imgfiles_list.append(file_list)
    roidb = []
    index = 0
    submit_dir = os.path.join(default.dataset_path, 'submit')
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    img_dir = os.path.join(default.dataset_path, 'test_result_img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for im in range(len(imgfiles_list)):
        index = im + 1;
        img_path = os.path.join(default.dataset_path, 'ch4_test_images','img_' + str(index) + '.jpg')
        img_ori = cv2.imread(img_path)
        batch = OneDataBatch(img_ori)
        mod.forward(batch, False)
        results = mod.get_outputs()
        output = dict(zip(mod.output_names, results))
        rois = output['rois_output'].asnumpy()[:, 1:]
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        mask_output = output['mask_prob_output'].asnumpy()
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, [img_ori.shape[0],img_ori.shape[1]])
        nms = py_nms_wrapper(config.TEST.NMS)
        boxes= pred_boxes
        CLASSES  = ('__background__', 'text')
        all_boxes = [[[] for _ in xrange(1)]
                     for _ in xrange(len(CLASSES))]
        all_masks = [[[] for _ in xrange(1)]
                     for _ in xrange(len(CLASSES))]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]
        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
        #print cls_scores.shape, label.shape
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep_la = nms(dets)
            print('------------------------keep_la',keep_la)
            all_boxes[cls_ind] = dets[keep_la, :]
            all_masks[cls_ind] = cls_masks[keep_la, :, :]
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]
        masks_this_image = [[]] + [all_masks[j] for j in range(1, len(CLASSES))]
        import copy
        import random
        class_names = CLASSES
        color_white = (255, 255, 255)
        scale = 1.0
        im = copy.copy(img_ori)
        num_box = 1
        num_boxes = 0
        mini_box = np.zeros((4,2))
        mini_box = np.int32(mini_box)
        if(len(dets) == 0):
            submit_path = os.path.join(submit_dir,'res_img_{}.txt'.format(index))
            result_txt = open(submit_path,'a')
            for i in range(0,4):
                result_txt.write(str(mini_box[i][0]))
                result_txt.write(',')
                result_txt.write(str(mini_box[i][1]))
                if i < 3:
                    result_txt.write(',')
            result_txt.write('\r\n')
            result_txt.close()
        for k, name in enumerate(class_names):
            if name == '__background__':
                continue
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
            dets = boxes_this_image[k]
            masks = masks_this_image[k]
            #im_binary_merge = np.zeros(im[:,:,0].shape)
            print('------------------------len(dets)',len(dets))
            for i in range(len(dets)):
                bbox_i = dets[i, :4] * scale
                #if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]  :
                if bbox_i[2] == bbox_i[0] or bbox_i[3] == bbox_i[1] :
                    continue
                score_i = dets[i, -1]
                bbox_i = map(int, bbox_i)
                mask_i = masks[i, :, :]
                mask_i = masks[i, :, :]
                mask_i = cv2.resize(mask_i, (bbox_i[2] - bbox_i[0], (bbox_i[3] - bbox_i[1])), interpolation=cv2.INTER_LINEAR)
                mask_i[mask_i > 0.3] = 1
                mask_i[mask_i <= 0.3] = 0
                im_binary_i = np.zeros(im[:,:,0].shape)
                im_binary_i[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2]] = im_binary_i[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2]] + mask_i
            #print("len(dets is )-------------------------",len(dets))
                overlap = []
                overlap_other = []
                for j in range(len(dets)):
                    if i == j:
                        continue
                    bbox_j = dets[j, :4] * scale
                #if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]  :
                    if bbox_j[2] == bbox_j[0] or bbox_j[3] == bbox_j[1] :
                        continue
                    num_box += 1
                    score_j = dets[j, -1]
                    bbox_j = map(int, bbox_j)
                    mask_j = masks[j, :, :]
                    mask_j = masks[j, :, :]
                    mask_j = cv2.resize(mask_j, (bbox_j[2] - bbox_j[0], (bbox_j[3] - bbox_j[1])), interpolation=cv2.INTER_LINEAR)
                    #print("mask_j,score_j,img_path------------------------",mask_j,score_j,img_path)
                    mask_j[mask_j > 0.3] = 1
                    mask_j[mask_j <= 0.3] = 0
                    im_binary_j = np.zeros(im[:,:,0].shape)
                    im_binary_j[bbox_j[1]: bbox_j[3], bbox_j[0]: bbox_j[2]] = im_binary_j[bbox_j[1]: bbox_j[3], bbox_j[0]: bbox_j[2]] + mask_j
                    im_binary = im_binary_i + im_binary_j
                    #mask_inter = mask_i+mask_j
                    ni = np.sum(im_binary_i == 1)
                    nj = np.sum(im_binary_j == 1)
                    nij = np.sum(im_binary==2)
                    IOU_ratio = float(nij)/(ni+nj-nij)
                    overlap.append(IOU_ratio)
                    #if np.sum(im_binary_i == 1) == 0:
                    #  continue
                    #if np.sum(im_binary_j == 1) == 0:
                    #  continue
                    IOU_ratio_self = float(np.sum(im_binary == 2)) / np.sum(im_binary_i == 1)
                    overlap_other.append(IOU_ratio_self)
                    #IOU_ratio_other = float(np.sum(im_binary == 2)) / np.sum(im_binary_j == 1)
                    #overlap_other.append(IOU_ratio_other)

                if num_box == 1:
                    overlap.append(0)
                    overlap_other.append(0)
                if np.max(overlap) < 0.6 and split == False and np.max(overlap_other) < 0.9:
                    num_boxes += 1
                    #cv2.rectangle(im, (bbox_i[0], bbox_i[1]), (bbox_i[2], bbox_i[3]), color=color, thickness=2)
                    cv2.putText(im, '%s %.3f' % (class_names[k], score_i), (bbox_i[0], bbox_i[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
                    px = np.where(mask_i == 1)
                    x_min = np.min(px[1])
                    y_min = np.min(px[0])
                    x_max = np.max(px[1])
                    y_max = np.max(px[0])
                    if x_max - x_min <= 1 or y_max - y_min <= 1:
                        continue
                    mask_color = random.randint(0, 255)
                    c = random.randint(0, 2)
                    mini_boxt = np.zeros((4,2))
                    target = im[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2], c] + mask_color * mask_i
                    target[target >= 255] = 255
                    im[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2], c] = target
                    mini_box = minimum_bounding_rectangle(im_binary_i)
                    mini_boxt[0][0] = mini_box[0][1]
                    mini_boxt[0][1] = mini_box[0][0]
                    mini_boxt[1][0] = mini_box[1][1]
                    mini_boxt[1][1] = mini_box[1][0]
                    mini_boxt[2][0] = mini_box[2][1]
                    mini_boxt[2][1] = mini_box[2][0]
                    mini_boxt[3][0] = mini_box[3][1]
                    mini_boxt[3][1] = mini_box[3][0]
                    mini_box = mini_boxt
                    mini_box = np.int32(mini_box)
                    #print("---------------",mini_box)
                    cv2.polylines(im, [mini_box],  1, (255,255,255))
                    submit_path = os.path.join(submit_dir,'res_img_{}.txt'.format(index))
                    result_txt = open(submit_path,'a')
                    for i in range(0,4):
                        result_txt.write(str(mini_box[i][0]))
                        result_txt.write(',')
                        result_txt.write(str(mini_box[i][1]))
                        if i < 3:
                            result_txt.write(',')
                    result_txt.write('\r\n')
                    result_txt.close()
                if split == True:
                    if np.max(overlap_other) > 0.6:
                        W = bbox_j[2] - bbox_j[0]
                        H = bbox_j[3] - bbox_j[1]
                        bbox_i[2] = bbox_i[2] - W
                        bbox_i[3] = bbox_i[3] - H
                    num_boxes += 1
                    cv2.rectangle(im, (bbox_i[0], bbox_i[1]), (bbox_i[2], bbox_i[3]), color=color, thickness=2)
                    cv2.putText(im, '%s %.3f' % (class_names[k], score_i), (bbox_i[0], bbox_i[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
                    px = np.where(mask_i == 1)
                    x_min = np.min(px[1])
                    y_min = np.min(px[0])
                    x_max = np.max(px[1])
                    y_max = np.max(px[0])
                    if x_max - x_min <= 1 or y_max - y_min <= 1:
                        continue
                    mask_color = random.randint(0, 255)
                    c = random.randint(0, 2)
                    target = im[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2], c] + mask_color * mask_i
                    target[target >= 255] = 255
                    im[bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2], c] = target
                    #inst_path = os.path.join(inst_dir,'result_{}_{}.mat'.format(index,num_boxes))
                    #io.savemat(inst_path, {'Segmentation': im_binary_i})
            #numbox = open('data/boxnum.txt','a')
            #numbox.write(str(num_boxes)+'\n')
            #numbox.close()
            result_img_path = os.path.join(img_dir,'result_{}.jpg'.format(index))
            cv2.imwrite(result_img_path,im)
    #zip_submit_dir = 'script_test_ch4'
    zip_file = os.path.join('script_test_ch4', 'submit.zip')
    createZip(submit_dir, zip_file)
    os.system("python ./script_test_ch4/script.py -g=./script_test_ch4/gt.zip -s=./script_test_ch4/submit.zip")

def createZip(filePath,savePath):
    newZip = zipfile.ZipFile(savePath,'w')
    fileList = []
    for dirpath,dirnames,filenames in os.walk(filePath):
        for filename in filenames:
            fileList.append(os.path.join(dirpath,filename))
    for tar in fileList:
        newZip.write(tar,tar[len(filePath):])
    newZip.close()
    print('zip successful')


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)    
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--image_name', help='image file path',type=str)
    
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    print args
    demo_maskrcnn(network = args.network, 
                  ctx = ctx,
                  prefix = args.prefix,
                  epoch = args.epoch, 
                  vis= args.vis, 
                  has_rpn = True,
                  thresh = args.thresh)

if __name__ == '__main__':
    main()

                
