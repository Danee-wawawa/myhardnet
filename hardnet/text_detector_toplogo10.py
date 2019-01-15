import copy
import random
import argparse
from rcnn.config import default, generate_config
from rcnn.symbol import *
from rcnn.utils.load_model import load_param
from rcnn.core.module import MutableModule
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper, gpu_nms_wrapper
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

MERGE_THRESH = 1.1 #no merge

class OneDataBatch():
    def __init__(self,img):
        im_info = mx.nd.array([[img.shape[0],img.shape[1],1.0]])
        img = np.transpose(img,(2,0,1)) 
        img = img[np.newaxis,(2,1,0)]
        self.data = [mx.nd.array(img),im_info]
        self.label = None
        self.provide_label = None
        self.provide_data = [("data",(1,3,img.shape[2],img.shape[3])),("im_info",(1,3))]

class TextDetector:
  def __init__(self, network, prefix, epoch, ctx_id=0, mask_nms=True):
    self.ctx_id = ctx_id
    self.ctx = mx.gpu(self.ctx_id)
    self.mask_nms = mask_nms
    #self.nms_threshold = 0.3
    #self._bbox_pred = nonlinear_pred
    if not self.mask_nms:
      self.nms = gpu_nms_wrapper(config.TEST.NMS, self.ctx_id)
    else:
      self.nms = gpu_nms_wrapper(config.TEST.RPN_NMS_THRESH, self.ctx_id)
    #self.nms = py_nms_wrapper(config.TEST.NMS)

    sym = eval('get_' + network + '_mask_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    #arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=self.ctx, process=True)
    split = False
    max_image_shape = (1,3,1600,1600)
    #max_image_shape = (1,3,1200,2200)
    max_data_shapes = [("data",max_image_shape),("im_info",(1,3))]
    mod = MutableModule(symbol = sym, data_names = ["data","im_info"], label_names= None,
                            max_data_shapes = max_data_shapes, context=self.ctx)
    mod.bind(data_shapes = max_data_shapes, label_shapes = None, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    self.model = mod
    pass

  def detect(self, img, scales=[1.], thresh=0.5):
    ret = []
    #scale = scales[0]
    dets_all = None
    masks_all = None
    for scale in scales:
      if scale!=1.0:
        nimg = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
      else:
        nimg = img
      im_size = nimg.shape[0:2]
      #im_info = mx.nd.array([[nimg.shape[0],nimg.shape[1],1.0]])
      #nimg = np.transpose(nimg,(2,0,1)) 
      #nimg = nimg[np.newaxis,(2,1,0)]
      #nimg = mx.nd.array(nimg)
      #db = mx.io.DataBatch(data=(nimg,im_info))
      db = OneDataBatch(nimg)
      self.model.forward(db, is_train=False)
      results = self.model.get_outputs()
      output = dict(zip(self.model.output_names, results))
      rois = output['rois_output'].asnumpy()[:, 1:]
      scores = output['cls_prob_reshape_output'].asnumpy()[0]
      bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
      mask_output = output['mask_prob_output'].asnumpy()
      pred_boxes = bbox_pred(rois, bbox_deltas)
      pred_boxes = clip_boxes(pred_boxes, [im_size[0],im_size[1]])
      boxes= pred_boxes
      label = np.argmax(scores, axis=1)
      label = label[:, np.newaxis]
      cls_ind = 1 #text class
      cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)] / scale
      cls_masks = mask_output[:, cls_ind, :, :]
      cls_scores = scores[:, cls_ind, np.newaxis]
      #print cls_scores.shape, label.shape
      keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
      dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
      masks = cls_masks[keep, :, :]
      if dets.shape[0]==0:
        continue
      if dets_all is None:
        dets_all = dets
        masks_all = masks
      else:
        dets_all = np.vstack((dets_all, dets))
        masks_all = np.vstack((masks_all, masks))
      #scores = dets[:,4]
      #index = np.argsort(scores)[::-1]
      #dets = dets[index]
      #print(dets)
    if dets_all is None:
      return np.zeros( (0,2) )
    dets = dets_all
    masks = masks_all

    keep = self.nms(dets)
    dets = dets[keep, :]
    masks = masks[keep, :, :]

    det_mask = np.zeros( (dets.shape[0],)+img.shape[0:2], dtype=np.int )
    mask_n = np.zeros( (dets.shape[0],), dtype=np.int )
    invalid = np.zeros( (dets.shape[0],), dtype=np.int )
    for i in range(dets.shape[0]):
        bbox_i = dets[i, :4]
        #if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]  :
        if bbox_i[2] == bbox_i[0] or bbox_i[3] == bbox_i[1] :
          invalid[i] = 1
          continue
        score_i = dets[i, -1]
        #bbox_i = map(int, bbox_i)
        bbox_i = bbox_i.astype(np.int)
        mask_i = masks[i, :, :]
        mask_i = cv2.resize(mask_i, (bbox_i[2] - bbox_i[0], (bbox_i[3] - bbox_i[1])), interpolation=cv2.INTER_LINEAR)
        #avg_mask = np.mean(mask_i[mask_i>0.5])
        #print('det', i, 'mask avg', avg_mask)
        mask_i[mask_i > 0.5] = 1
        mask_i[mask_i <= 0.5] = 0
        det_mask[i, bbox_i[1]: bbox_i[3], bbox_i[0]: bbox_i[2]] += mask_i.astype(np.int)
        mask_n[i] = np.sum(mask_i==1)

    if self.mask_nms:
      for i in range(dets.shape[0]):
        if invalid[i]>0:
          continue
        mask_i = det_mask[i]
        ni = mask_n[i]
        merge_list = []
        for j in range(i+1,  dets.shape[0]):
          if invalid[j]>0:
            continue
          mask_j = det_mask[j]
          nj = mask_n[j]
          mask_inter = mask_i+mask_j
          nij = np.sum(mask_inter==2)
          iou = float(nij)/(ni+nj-nij)
          iou_i = float(nij)/ni
          iou_j = float(nij)/nj
          if iou_j > 0.7:
            invalid[j] = 1
          if iou>=config.TEST.NMS:
          #if iou>=0.7:
            invalid[j] = 1
            if iou>=MERGE_THRESH:
              merge_list.append(j)
              #mask_i = np.logical_or(mask_i, mask_j, dtype=np.int).astype(np.int)
              #det_mask[i] = mask_i
              #print(mask_i)
        for mm in merge_list:
          _mask = det_mask[mm]
          mask_i = np.logical_or(mask_i, _mask, dtype=np.int)
        if len(merge_list)>0:
          det_mask[i] = mask_i.astype(np.int)

    for i in range(dets.shape[0]):
      if invalid[i]>0:
        continue
      mask_i = det_mask[i]

      mini_box = minimum_bounding_rectangle(mask_i)
      mini_boxt = np.zeros((4,2))
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
      ret.append(mini_box)
      #scores.append(score_i)
      #print("---------------",mini_box)
      #cv2.polylines(im, [mini_box],  1, (255,255,255))
      #submit_path = os.path.join(submit_dir,'res_img_{}.txt'.format(index))
      #result_txt = open(submit_path,'a')
      #for i in range(0,4):
      #    result_txt.write(str(mini_box[i][0]))
      #    result_txt.write(',')
      #    result_txt.write(str(mini_box[i][1]))
      #    if i < 3:
      #        result_txt.write(',')
      #result_txt.write('\r\n')
      #result_txt.close()
    return ret

