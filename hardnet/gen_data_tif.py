import os
import glob
import sys
import cv2
from scipy import io
import numpy as np
import PIL.Image as Image
#reload(sys)
#sys.setdefaultencoding('utf-8')
dataset_root = "/gpu/data2/icdar2015"
seg_dir = 'seg_label'
lst_dir = 'imglists'

for _dir in [lst_dir, seg_dir]:
  __dir = os.path.join(dataset_root, _dir)
  if not os.path.exists(__dir):
    os.mkdir(__dir)


for image_set in ['train', 'test']:

  lst = open(os.path.join(dataset_root, lst_dir, '%s.lst'%image_set), 'w')
  if image_set=='test':
    image_dir_name = 'ch4_%s_images' % image_set
    gt_dir = os.path.join(dataset_root, 'ch4_test_gt','ch4_test_gt')
  else:
    image_dir_name = 'ch4_training_images'
    gt_dir = os.path.join(dataset_root, 'ch4_training_localization_transcription_gt')
  image_dir = os.path.join(dataset_root, image_dir_name)
  image_paths = glob.glob(os.path.join(image_dir,'*.jpg'))
  gt_paths = [os.path.join(gt_dir, 'gt_{}.txt'.format(os.path.basename(o)[:-4])) for o in image_paths]
  n_samples = len(image_paths)
  for imgid in range(n_samples):
    image_path = image_paths[imgid]
    w_image_path = '/'.join(image_path.split('/')[-2:])
    #print(imgid, image_path)
    debug = False

   # print(os.os.path.join(image_dir, '{}.jpg'.format(os.path.basename(o)[:-4]))
    #size = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(os.path.basename(o)[:-4]))).shape
    size = cv2.imread(image_path).shape
    height = size[0]
    width = size[1]
    #print(width)
    #print(height)
    #gt_path = os.path.join(gt_dir, 'gt_{}.txt'.format(os.path.basename(o)[:-4]))
    #gt_path = ''.join(gt_path)
    #gt_path = gt_path.decode('utf-8')
    gt_path = gt_paths[imgid]
    if image_set=='train' and image_path.find('img_88.jpg')>=0:
      debug = True
      print(w_image_path, gt_path)
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
      if debug:
        print('line', line)
      num_segclass = num_segclass + 1
      #print(line)
      splits = line.split(',')
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
    seg_file_name = os.path.join(seg_dir, '%s_%d.tif'%(image_set, imgid))
    seg_file = os.path.join(dataset_root, seg_file_name)
    seg_image = Image.fromarray(inst)
    seg_image.save(seg_file)
    lst.write("%d\t%s\t%s\n"%(imgid, w_image_path, seg_file_name))
    if debug:
      pixel = inst
      class_id = [0, 1]
      boxes = []
      gt_classes = []
      ins_id = []
      gt_overlaps = []
      for c in range(1, len(class_id)):
        px = np.where((pixel >= class_id[c] * 1000) & (pixel < (class_id[c] + 1) * 1000))
        if len(px[0]) == 0:
            continue
        ids = np.unique(pixel[px])
        for id in ids:
            px = np.where(pixel == id)
            x_min = np.min(px[1])
            y_min = np.min(px[0])
            x_max = np.max(px[1])
            y_max = np.max(px[0])
            if x_max - x_min <= 1 or y_max - y_min <= 1:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
      print(boxes)
    #inst_dir = os.path.join(data_root, 'test_inst')
    #if not os.path.exists(inst_dir):
    #  os.makedirs(inst_dir)
    #inst_path = os.path.join(inst_dir, '{}.mat'.format(os.path.basename(o)[:-4]))
    #io.savemat(inst_path, {'Segmentation': inst})
    #cls_dir = os.path.join(data_root, 'test_cls')
    #if not os.path.exists(cls_dir):
    #  os.makedirs(cls_dir)
    #cls_path = os.path.join(cls_dir, '{}.mat'.format(os.path.basename(o)[:-4]))
    #io.savemat(cls_path, {'Segmentation': cls})
    #ind = np.where(cls == 1)[0]
    #print(ind)  #print(polygon)
  lst.close()
#data_root = '/gpu/data2/icdar2015/cls'
#data_dir = os.path.join(data_root,'img_9.mat')
#seg = sio.loadmat(data_dir)
#print(seg)
#seg_data = seg['Segmentation']
#print(seg_data)
#ind = np.where(seg_data == 1)[0]
#print(ind)
sys.exit(0)
data_dir = '/gpu/data2/icdar2015/ch4_test_images_rename/'
filenames = os.listdir(data_dir)
#print(filenames)
for name in filenames:
  filenames[filenames.index(name)] = name[:-4]
out = open('val.txt','w')
for name in filenames:
  out.write(name+'\n')
out.close()
