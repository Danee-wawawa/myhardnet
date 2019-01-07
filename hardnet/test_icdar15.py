import argparse
from rcnn.config import default, generate_config

import numpy as np
import os
from scipy import io
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile
from text_detector import TextDetector

def test(network, ctx_id, prefix, epoch, name,
                   vis= True, has_rpn = True, 
                   thresh = 0.001, scale='960'):
    
    assert has_rpn,"Only has_rpn==True has been supported."
    detector = TextDetector(network, prefix, epoch, ctx_id)

    #imglist_file = os.path.join(default.dataset_path, 'imglists', 'test.lst')
    #assert os.path.exists(imglist_file), 'Path does not exist: {}'.format(imglist_file)
    #imgfiles_list = []
    #with open(imglist_file, 'r') as f:
    #    for line in f:
    #        file_list = dict()
    #        label = line.strip().split('\t')
    #        file_list['img_path'] = label[1]
    #        imgfiles_list.append(file_list)
    roidb = []
    index = 0
    results = []
    scores = []
    submit_root = './icdar15_submit'
    submit_dir = os.path.join(submit_root, name)
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    img_dir = os.path.join(submit_root, '%s_images'%name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    origin_im_size = 720
    S = [int(x) for x in scale.split(',')]
    scales = [float(x)/origin_im_size for x in S]
    print('test with scales', scales)
    for idx in range(500):
        index = idx + 1;
        img_path = os.path.join(default.dataset_path, 'ch4_test_images','img_' + str(index) + '.jpg')

        test_gt_path = os.path.join(default.dataset_path, 'ch4_test_gt','ch4_test_gt','gt_img_' + str(index) + '.txt')

        im = cv2.imread(img_path)
        submit_path = os.path.join(submit_dir,'res_img_{}.txt'.format(index))
        result_txt = open(submit_path,'w')
        results = detector.detect(im, thresh=thresh, scales=scales)
        num_text = len(results)
        print(img_path, num_text)
        #box_index = 0
        for mini_box in results:
          #print("---------------",mini_box)
          cv2.polylines(im, [mini_box],  1, (0,0,255))
          #cv2.putText(im, 'Text %.3f' % (scores[box_index]), (mini_box[0][0], mini_box[0][1] + 10),
          #            color=(255,255,255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
          #box_index += 1
          for i in range(0,4):
              result_txt.write(str(mini_box[i][0]))
              result_txt.write(',')
              result_txt.write(str(mini_box[i][1]))
              if i < 3:
                  result_txt.write(',')
          result_txt.write('\n')
          #numbox = open('data/boxnum.txt','a')
          #numbox.write(str(num_boxes)+'\n')
          #numbox.close()
        result_txt.close()
        lines = []
        with open(test_gt_path, 'r') as f:
            for line in f.readlines():
                if '\xef\xbb\xbf'  in line:
                    str1 = line.replace('\xef\xbb\xbf','')
                    lines.append(str1.strip())
                else:
                    lines.append(line.strip())
        for line in lines:
            splits = line.split(',')
            gt_box = np.zeros((4,2))
            gt_box[0][0] = int(splits[0])
            gt_box[0][1] = int(splits[1])
            gt_box[1][0] = int(splits[2])
            gt_box[1][1] = int(splits[3])
            gt_box[2][0] = int(splits[4])
            gt_box[2][1] = int(splits[5])
            gt_box[3][0] = int(splits[6])
            gt_box[3][1] = int(splits[7])
            gt_box = np.int32(gt_box)
            cv2.polylines(im, [gt_box],  1, (0,255,0))
        result_img_path = os.path.join(img_dir,'result_{}_{}.jpg'.format(index, num_text))
        cv2.imwrite(result_img_path,im)
    #zip_submit_dir = 'script_test_ch4'
    zip_file = os.path.join(submit_root, '%s.zip'%name)
    if os.path.exists(zip_file):
      os.remove(zip_file)
    createZip(submit_dir, zip_file)
    cmd = "python %s/script.py -g=%s/gt.zip -s=%s"%(default.dataset_path, default.dataset_path, zip_file)
    print(cmd)
    os.system(cmd)

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
    parser.add_argument('--epoch', help='model to test with', default=0, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    parser.add_argument('--name', help='test name', default='submit', type=str)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=0.3, type=float)
    parser.add_argument('--scale', help='test image size', default='960', type=str)
    
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print args
    test(network = args.network, 
                  ctx_id = args.gpu,
                  prefix = args.prefix,
                  epoch = args.epoch, 
                  name = args.name,
                  vis= args.vis, 
                  has_rpn = True,
                  thresh = args.thresh,
                  scale = args.scale)

if __name__ == '__main__':
    main()

                
