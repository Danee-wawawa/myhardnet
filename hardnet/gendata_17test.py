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
import zipfile
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
save_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/1030/Text-Detection/Text-Detection/ic17-mlt"
lst_dir = 'imglists_17'
image_dir_name = 'val'
gt_dir = os.path.join(save_root,'label','val')
n_samples = 1800
image_dir = os.path.join(save_root, 'image', image_dir_name)
test_lst = open(os.path.join(dataset_root,lst_dir,'test.lst'),'w')
for imgid in range(1, n_samples+1):
  image_path = os.path.join(image_dir, 'img_%d.jpg'%imgid)
  if not os.path.exists(image_path):
    print '---------------------img_path',image_path
  else:
    print imgid, image_path
  test_lst.write("%d\t%s\n"%(imgid, image_path))
test_lst.close()
#gt_path = os.path.join(gt_dir, 'gt_img_%d.txt'%imgid)
submit_root = '../script_test_ch4'
name = 'gt'
zip_file = os.path.join(submit_root, '%s.zip'%name)
submit_dir = '/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/1030/Text-Detection/Text-Detection/ic17-mlt/label/val'
if os.path.exists(zip_file):
  os.remove(zip_file)
#createZip(submit_dir, zip_file)

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
createZip(submit_dir, zip_file)

