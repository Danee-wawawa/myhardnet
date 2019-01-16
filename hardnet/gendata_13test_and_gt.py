import cv2
import os
import numpy as np
import PIL.Image as Image
import zipfile
save_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/1030/Text-Detection/Text-Detection"
txt_dir = 'image_list'
lst_dir = 'imglists_13'
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
#for filedir in [gt_dir]:
#    filedir = os.path.join(save_root,'ic13',filedir)
#    if not os.path.exists(filedir):
#        os.mkdir(filedir)
test_lst = open(os.path.join(save_root,lst_dir,'test.lst'),'w')
imgtxt_file = os.path.join(dataset_root, txt_dir, 'test_ic13.txt')
img_id = 0
with open(imgtxt_file, 'r') as f:
    for txtpath in f:
        img_id = img_id+1
        #if img_id == 392 or img_id == 379 or img_id == 383 or img_id == 389:
        #    continue
        #if img_id > 400:
        #    break
        labels = txtpath.strip().split('\t')
        #print '--------------------',label
        label = str(labels[0])
        img_path = label.split(' ')[0]
        gt_path = label.split(' ')[1]
        lines = []
        img_path = os.path.join(dataset_root,img_path)
        if not os.path.exists(img_path):
            print '-----------------img_path no exits',img_path
            continue
        print '--------------------img_path',img_path
        test_lst.write("%d\t%s\n"%(img_id, img_path))
    test_lst.close()
#gt_path = os.path.join(dataset_root, 'ic13', 'Challenge1_Test_Task1_GT')
submit_root = '../script_test_ch4'
name = 'gt'
zip_file = os.path.join(submit_root, '%s.zip'%name)
submit_dir = os.path.join(dataset_root, 'ic13', 'Challenge1_Test_Task1_GT')
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

        