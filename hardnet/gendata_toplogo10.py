import cv2
import os
import numpy as np
import PIL.Image as Image
import zipfile
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
seg_dir = 'seg_labels_toplogo10'
lst_dir = 'imglists_toplogo10'
gt_dir = 'gt'
dataset = 'toplogo10'
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
gtdir = os.path.join(dataset_root,dataset,gt_dir)
if not os.path.exists(gtdir):
	os.mkdir(gtdir)
for filedir in [seg_dir,lst_dir]:
    filedir = os.path.join(dataset_root,filedir)
    if not os.path.exists(filedir):
        os.mkdir(filedir)
train_lst = open(os.path.join(dataset_root,lst_dir,'train.lst'),'w')
img_num = 0
for logoname in ['adidas','chanel','Gucci','HH','lacoste','MK','nike','prada','puma','supreme']:
    filedir = os.path.join(dataset_root,'toplogo10','jpg',logoname)
    for num in range(1,61):
    	img_num = img_num+1
    	img_path = os.path.join(dataset_root,'toplogo10','jpg',logoname,logoname+str(num)+'.jpg')
        print '------------------------------------img_path',img_path
    	if not os.path.exists(img_path):
            print '------------------------img_path not exist'
    	im = cv2.imread(img_path)
    	height = im.shape[0]
    	width = im.shape[1]
        txt_path = os.path.join(dataset_root,'toplogo10','masks',logoname,logoname+str(num)+'.jpg.bboxes.txt')
        if not os.path.exists(txt_path):
            print '------------------------txt_path not exist'  
        txtfile_lines = []
        mask_final = np.zeros((height,width))
        seg_id = 1000
        with open(txt_path, 'r') as txt_file:
            for txtfile_line in txt_file.readlines():
                if '\xef\xbb\xbf'  in txtfile_line:
                    str1 = txtfile_line.replace('\xef\xbb\xbf','')
                    txtfile_lines.append(str1.strip())
                else:
                    txtfile_lines.append(txtfile_line.strip())
            for txtfile_line in txtfile_lines:
            	seg_id = seg_id+1
                #print '-----------------',txtfile_line
            	splits = txtfile_line.strip().split(' ')
            	mask_final[int(splits[1]): int(splits[1]) + int(splits[3]), int(splits[0]): int(splits[0]) + int(splits[2])]=seg_id
                #print '----------------splits',splits[1]
            seg_label_name = os.path.join(seg_dir,'train_%s_%s.jpg'%(logoname,num))
            seg_path = os.path.join(dataset_root,seg_label_name)
            seg_image = Image.fromarray(mask_final)
            seg_image = seg_image.convert('RGB')
            #seg_label_name = os.path.join(seg_dir,'train_%s_%s.tif'%(logoname,num))
            #seg_path = os.path.join(dataset_root,seg_label_name)
            #seg_image = Image.fromarray(mask_final)
            seg_image.save(seg_path)
            train_lst.write("%d\t%s\t%s\n"%(img_num, img_path, seg_path))
train_lst.close()


test_lst = open(os.path.join(dataset_root,lst_dir,'test.lst'),'w')
img_num = 0
for logoname in ['adidas','chanel','Gucci','HH','lacoste','MK','nike','prada','puma','supreme']:
    filedir = os.path.join(dataset_root,'toplogo10','jpg',logoname)
    for num in range(61,71):
    	img_num = img_num+1
    	img_path = os.path.join(dataset_root,'toplogo10','jpg',logoname,logoname+str(num)+'.jpg')
        print '-----------',img_path
        test_lst.write("%d\t%s\n"%(img_num, img_path))
        gt_txt = open(os.path.join(dataset_root,dataset,gt_dir,'gt_img_' + str(img_num) + '.txt'),'w')
        txt_path = os.path.join(dataset_root,'toplogo10','masks',logoname,logoname+str(num)+'.jpg.bboxes.txt')
        if not os.path.exists(txt_path):
            print '------------------------txt_path not exist'  
        txtfile_lines = []
        with open(txt_path, 'r') as txt_file:
            for txtfile_line in txt_file.readlines():
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
test_lst.close()
submit_root = '../script_test_ch4'
name = 'gt'
zip_file = os.path.join(submit_root, '%s.zip'%name)
submit_dir = '/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/toplogo10/gt'
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
