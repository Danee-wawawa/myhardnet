import cv2
import os
import numpy as np
import PIL.Image as Image
import zipfile
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
#seg_dir = 'seg_labels_flickr32'
lst_dir = 'imglists_flickr32'
gt_dir = 'gt'
dataset = 'FlickrLogos-v2'
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
#for filedir in [seg_dir,lst_dir]:
gtdir = os.path.join(dataset_root,dataset,gt_dir)
if not os.path.exists(gtdir):
	os.mkdir(gtdir)
imglist_file = os.path.join(dataset_root, lst_dir, 'test.lst')
#print(default.dataset_path)
assert os.path.exists(imglist_file), 'Path does not exist: {}'.format(imglist_file)
imgfiles_list = []
lines = []
index = 0
with open(imglist_file, 'r') as f:
    for line in f.readlines():
        if '\xef\xbb\xbf'  in line:
            str1 = line.replace('\xef\xbb\xbf','')
            lines.append(str1.strip())
        else:
            lines.append(line.strip())
    for line in lines:
    	index = index+1
        img_paths = line.strip().split('\t')
        img_path = str(img_paths[1])
        print '----------------------img_path',img_path
        img_id = img_path.strip().split('/')[-1:]
        img_id = str(img_id[0])
        #print '++++++++++',img_id
        img_id_path = '/'.join(img_path.split('/')[-1:])
        grand = img_path.strip().split('/')[-2:-1]
        grand = str(grand[0])
        grand_path = '/'.join(img_path.split('/')[-2:-1])
        txt_path = os.path.join(dataset_root, 'FlickrLogos-v2','classes','masks')
        txt_path = os.path.join(txt_path, grand_path)
        txt_path = os.path.join(txt_path, img_id + '.bboxes.txt')
        gt_txt = open(os.path.join(dataset_root,dataset,gt_dir,'gt_img_' + str(index) + '.txt'),'w')
        txtfile_lines = []
        with open(txt_path, 'r') as txt_file:
            for txtfile_line in txt_file.readlines()[1:]:
                if '\xef\xbb\xbf'  in txtfile_line:
                    str1 = txtfile_line.replace('\xef\xbb\xbf','')
                    txtfile_lines.append(str1.strip())
                else:
                    txtfile_lines.append(txtfile_line.strip())
            for txtfile_line in txtfile_lines:
                #print '-----------------',txtfile_line
            	splits = txtfile_line.strip().split(' ')
                #print '----------------splits',splits[1]
            	gt_box = np.zeros((4,2),dtype=np.int)
                gt_box[0][0] = int(splits[0])
                gt_box[0][1] = int(splits[1])
                gt_box[1][0] = int(splits[0]) + int(splits[2])
                gt_box[1][1] = int(splits[1])
                gt_box[2][0] = int(splits[0]) + int(splits[2])
                gt_box[2][1] = int(splits[1]) + int(splits[3])
                gt_box[3][0] = int(splits[0])
                gt_box[3][1] = int(splits[1]) + int(splits[3])
                #print '--------------',gt_box 
                for i in range(0,4):
                    gt_txt.write(str(gt_box[i][0]))
                    gt_txt.write(',')
                    gt_txt.write(str(gt_box[i][1]))
                    gt_txt.write(',')
                    if i == 3:
                    	gt_txt.write('###')
                gt_txt.write('\n')
            gt_txt.close()
submit_root = '../script_test_ch4'
name = 'gt'
zip_file = os.path.join(submit_root, '%s.zip'%name)
submit_dir = '/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/FlickrLogos-v2/gt'
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
