from __future__ import division
import os
import glob
import sys
import random
import cv2
import shutil
from scipy import io
import numpy as np
import PIL.Image as Image


SIZE = (1280, 720)
DEBUG = False
NUM_CROP = 10
LIMIT = 200
CROP_SIZE = 640
rotates=[0, 15, 30, 45]
CROP_ROTATE = False



def get_crop_image(im_image, inst_image):
  #SIZE = config.CROP_SCALES[0][0]
  #CROP_SIZE = 640
  #PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
  PRE_SCALES = [0.3, 0.5, 0.7, 1.0, 1.5]
  _scale = random.choice(PRE_SCALES)
  im = np.asarray(im_image)
  inst = np.asarray(inst_image)
  size = int(np.round(_scale*np.min(im.shape[0:2])))
  im_scale = float(CROP_SIZE)/size
  origin_shape = im.shape
  if _scale>1.0:
    sizex = int(np.round(im.shape[1]*im_scale))
    sizey = int(np.round(im.shape[0]*im_scale))
    if sizex<CROP_SIZE:
      sizex = CROP_SIZE
      print('keepx',sizex)
    if sizey<CROP_SIZE:
      sizey = CROP_SIZE
      print('keepy',sizey)
    im = cv2.resize(im,(sizex,sizey),interpolation=cv2.INTER_LINEAR)
  else:
    im = cv2.resize(im,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
    inst = cv2.resize(inst,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
  assert im.shape[0]>=CROP_SIZE and im.shape[1]>=CROP_SIZE
  assert inst.shape[0]>=CROP_SIZE and inst.shape[1]>=CROP_SIZE

  retry = 0
  #LIMIT = 25
  size = CROP_SIZE
  test_flag = True
  while retry<LIMIT:
    up, left = (np.random.randint(0,im.shape[0]-size+1),np.random.randint(0,im.shape[1]-size+1))
    im_new = im[up:(up+size), left:(left+size), :]
    inst_new = inst[up:(up+size), left:(left+size)]
    #print("np.sum(inst_new>=1001)---------------------------------",np.sum(inst_new>=1001))
    if np.sum(inst_new>=1001) >= 2000 and np.sum(inst_new[0,0:CROP_SIZE]!=0)==0 \
    and np.sum(inst_new[CROP_SIZE-1,0:CROP_SIZE]!=0)==0 and np.sum(inst_new[0:CROP_SIZE,0]!=0)==0 \
    and np.sum(inst_new[0:CROP_SIZE,CROP_SIZE-1]!=0)==0:
      break
    if retry == LIMIT - 1:
      test_flag = False 
    retry += 1
  im_new_img = Image.fromarray(im_new)
  inst_new_img = Image.fromarray(inst_new)
  return test_flag, im_new_img, inst_new_img, _scale


def get_gt_seg(gt_path, height, width):
  lines = []
  with open(gt_path, 'r') as f:
    for line in f.readlines():
      if '\xef\xbb\xbf'  in line:
        str1 = line.replace('\xef\xbb\xbf','')
        lines.append(str1.strip())
      else:
        lines.append(line.strip())
   # lines = [m.strip() for m in f.readlines()]
 # print(lines)
  word_polygons = []
  words = []
  num_segclass = 1000
  inst = np.zeros((height, width), dtype=np.uint16)
  cls = np.zeros((height, width), dtype=np.uint16)
  for line in lines:
    num_segclass = num_segclass + 1
    #print(line)
    splits = line.split(',')
    if splits[8]=='###':
      #print('find ###')
      continue
    polygon = [float(int(n)) for n in splits[:8]]
    word = ','.join(splits[8:])
    word = bytes(word)
    words.append(word)
    A_x = int(splits[0])
    A_y = int(splits[1])
    B_x = int(splits[2])
    B_y = int(splits[3])
    C_x = int(splits[4])
    C_y = int(splits[5])
    D_x = int(splits[6])
    D_y = int(splits[7])
    x1 = int(min(A_x, B_x, C_x, D_x))
    x2 = int(max(A_x, B_x, C_x, D_x))
    y1 = int(min(A_y, B_y, C_y, D_y))
    y2 = int(max(A_y, B_y, C_y, D_y))
    for i in range(x1,x2):
      for j in range(y1,y2):
        a = (B_x - A_x)*(j - A_y) - (B_y - A_y)*(i - A_x)
        b = (C_x - B_x)*(j - B_y) - (C_y - B_y)*(i - B_x)
        c = (D_x - C_x)*(j - C_y) - (D_y - C_y)*(i - C_x)
        d = (A_x - D_x)*(j - D_y) - (A_y - D_y)*(i - D_x)
        if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
       # print(i)
          inst[j, i] = num_segclass
          cls[j, i] = 1
   # word_polygons.extend(polygon)
  #print(inst)
  if len(words)==0:
    return None
  inst = Image.fromarray(inst)
  return inst


def get_rotate(image, inst, angle):
  assert angle!=0
  img = image.rotate(angle, expand=True)
  seg = inst.rotate(angle, expand=True)
  seg = np.asarray(seg)
  img = np.asarray(img)                   
  for i in range(2):
    assert seg.shape[i]==img.shape[i]
  height = seg.shape[0]
  width = seg.shape[1]
  #print("img_matrix.shape------------------------",img_matrix.shape)
  px = np.where(seg>=1000)
  assert len(px[0])>0
  #print("----------------",px)
  #if len(px[0]) == 0:
  #    continue
  x_min = np.min(px[1])
  y_min = np.min(px[0])
  x_max = np.max(px[1])
  y_max = np.max(px[0])
  if x_max - x_min <= 1 or y_max - y_min <= 1:
    return None, None
  diffs = [0, -50, 50, -100, 100]
  scales = []
  for diff in diffs:
    center = [width//2, height//2]
    center[0] += diff
    center[1] += diff
    mwidth = max(np.abs(center[0]-x_min), np.abs(center[0]-x_max))+1
    mheight = max(np.abs(center[1]-y_min), np.abs(center[1]-y_max))+1
    mwidth*=2
    mheight*=2
    wscale = float(mwidth)/SIZE[0]
    hscale = float(mheight)/SIZE[1]
    scale = max(wscale, hscale, 1.0)
    scales.append(scale)
  sel = np.argmin(scales)
  diff = diffs[sel]
  scale = scales[sel]
  #print(scales, sel, diff, scale)
  #assert scale==1.0
  center = [width//2, height//2]
  center[0] += diff
  center[1] += diff
  M = [ 
        [scale, 0, SIZE[0]//2-center[0]*scale],
        [0, scale, SIZE[1]//2-center[1]*scale],
      ]
  M = np.array(M, dtype=np.float32)
  #print(M, IM)
  img = cv2.warpAffine(img, M, SIZE)
  seg = cv2.warpAffine(seg, M, SIZE, flags=cv2.INTER_NEAREST)
  img = Image.fromarray(img)
  seg = Image.fromarray(seg)
  return img, seg

dataset_root = "./"
seg_dir = 'seg_label_rotate_crop_test'
lst_dir = 'imglists_test'
rotate_image_dir = 'ch4_training_images_rotate_crop_test'

for _dir in [lst_dir, seg_dir, rotate_image_dir]:
  __dir = os.path.join(dataset_root, _dir)
  if not os.path.exists(__dir):
    os.mkdir(__dir)
  else:
    shutil.rmtree(_dir)
    os.mkdir(__dir)


for image_set in ['train', 'test']:
  lst = open(os.path.join(dataset_root, lst_dir, '%s.lst'%image_set), 'w')
  if image_set=='test':
    image_dir_name = 'ch4_%s_images' % image_set
    gt_dir = os.path.join(dataset_root, 'ch4_test_gt','ch4_test_gt')
    rotates=[0]
    n_samples = 500
  else:
    image_dir_name = 'ch4_training_images'
    gt_dir = os.path.join(dataset_root, 'ch4_training_localization_transcription_gt')
    rotates=[0, 15, 30, 45]
    n_samples = 1000
  assert rotates[0]==0
  image_dir = os.path.join(dataset_root, image_dir_name)
  idx=0
  for imgid in range(1, n_samples+1):
    image_path = os.path.join(image_dir, 'img_%d.jpg'%imgid)
    gt_path = os.path.join(gt_dir, 'gt_img_%d.txt'%imgid)
    print(imgid, image_path)
    w_image_path = '/'.join(image_path.split('/')[-2:])
    img = Image.open(image_path).convert('RGB')
    size = img.size
    height = size[1]
    width = size[0]
    inst = get_gt_seg(gt_path, height, width)
    #if inst is None:
    #  print('YYY error')
    #  continue
    if image_set == 'train': 
      if inst is None:
        print('YYY error')
        continue
      for angle in rotates:
        if angle==0:
        #seg_file_name = os.path.join(seg_dir, '%s_%d_%d.tif'%(image_set, imgid, angle))
        #seg_file = os.path.join(dataset_root, seg_file_name)
        #inst.save(seg_file)
        #lst.write("%d\t%s\t%s\n"%(idx, w_image_path, seg_file_name))
        #rimg = img
        #rinst = inst
          for i in range(NUM_CROP):
            Flag, rimg, rinst, crop_scale = get_crop_image(img, inst)
            if Flag == True: 
              #rinst = rinst.convert('RGB')
              seg_file_name = os.path.join(seg_dir, '%s_%d_%d_%.2f_%d.tif'%(image_set, imgid, angle, crop_scale, i))
            #seg_file_name = os.path.join(seg_dir, '%s_%d_%d_%.2f_%d.jpg'%(image_set, imgid, angle, crop_scale, i))
              seg_file = os.path.join(dataset_root, seg_file_name)
              rinst.save(seg_file)
              _image_path = os.path.join(rotate_image_dir, '%s_%d_%d_%.2f_%d.jpg'%(image_set, imgid, angle, crop_scale,i))
              rimg.save(_image_path)
              _w_image_path = '/'.join(_image_path.split('/')[-2:])
              lst.write("%d\t%s\t%s\n"%(idx, _w_image_path, seg_file_name))
              idx+=1
            else:
              i = i - 1
              continue
          #else:
      #  rimg, rinst = get_rotate(img, inst, angle)
      #  if rimg is None:
      #    print('XXX error')
      #    continue
        #seg_file_name = os.path.join(seg_dir, '%s_%d_%d.tif'%(image_set, imgid, angle))
        #seg_file = os.path.join(dataset_root, seg_file_name)
        #rinst.save(seg_file)
        #_image_path = os.path.join(rotate_image_dir, 'img_%d_%d.jpg'%(imgid, angle))
        #rimg.save(_image_path)
        #_w_image_path = '/'.join(_image_path.split('/')[-2:])
        #lst.write("%d\t%s\t%s\n"%(idx, _w_image_path, seg_file_name))
        
        if CROP_ROTATE:
          for i in range(NUM_CROP):
            rimg, rinst, crop_scale = get_crop_image(rimg, rinst)
            #rinst = rinst.convert('RGB')
            seg_file_name = os.path.join(seg_dir, '%s_%d_%d_%.2f_%d.tif'%(image_set, imgid, angle, crop_scale, i))
        #seg_file_name = os.path.join(seg_dir, '%s_%d_%d_%.2f.jpg'%(image_set, imgid, angle, crop_scale))
            seg_file = os.path.join(dataset_root, seg_file_name)
            rinst.save(seg_file)
            _image_path = os.path.join(rotate_image_dir, '%s_%d_%d_%.2f_%d.jpg'%(image_set,imgid, angle, crop_scale, i))
            rimg.save(_image_path)
            _w_image_path = '/'.join(_image_path.split('/')[-2:])
            lst.write("%d\t%s\t%s\n"%(idx, _w_image_path, seg_file_name))
            idx+=1
    else:
      seg_file_name = os.path.join(seg_dir, '%s_%d.tif'%(image_set, imgid))
      seg_file = os.path.join(dataset_root, seg_file_name)
      #inst.save(seg_file)
      #lst.write("%d\t%s\t%s\n"%(idx, w_image_path, seg_file_name))
      lst.write("%d\t%s\n"%(idx, w_image_path))
      idx+=1
  lst.close()

