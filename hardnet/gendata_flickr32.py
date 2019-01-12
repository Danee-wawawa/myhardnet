import cv2
import os
import numpy as np
import PIL.Image as Image
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
seg_dir = 'seg_labels_flickr32'
lst_dir = 'imglists_flickr32'
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
for filedir in [seg_dir,lst_dir]:
    filedir = os.path.join(dataset_root,filedir)
    if not os.path.exists(filedir):
        os.mkdir(filedir)
train_lst = open(os.path.join(dataset_root,lst_dir,'train.lst'),'w')
imgtxt_file = os.path.join(dataset_root, 'FlickrLogos-v2', 'trainvalset.relpaths.txt')
img_num = 0
lines = []
with open(imgtxt_file, 'r') as f:
    for line in f.readlines():
        if '\xef\xbb\xbf'  in line:
            str1 = line.replace('\xef\xbb\xbf','')
            lines.append(str1.strip())
        else:
            lines.append(line.strip())
    for line in lines:
        txtpath = str(line)
        
#with open(imgtxt_file, 'r') as f:
#    for txtpath in f:
        img_num = img_num+1
        if img_num > 1280:
            break
        img_path = os.path.join(dataset_root, 'FlickrLogos-v2', txtpath)
        #img_path = os.path.join(dataset_root, 'FlickrLogos-v2')
        #print '--------------------------',txtpath
        if not os.path.exists(img_path):
            print '------------------------img_path not exist'
        print '------------------img_path',img_path
        im = cv2.imread(img_path)
        img_id = img_path.strip().split('/')[-1:]
        img_id = str(img_id[0])
        #print '++++++++++',img_id
        img_id_path = '/'.join(img_path.split('/')[-1:])
        grand = img_path.strip().split('/')[-2:-1]
        grand = str(grand[0])
        grand_path = '/'.join(img_path.split('/')[-2:-1])
        #img_id = img_id.split('.')[0]
        txt_path = os.path.join(dataset_root, 'FlickrLogos-v2','classes','masks')
        txt_path = os.path.join(txt_path, grand_path)
        txt_path = os.path.join(txt_path, img_id + '.bboxes.txt')
        txt_file = open(txt_path, 'r')       
        lines = len(txt_file.readlines())
        lines = lines - 1
        seg_id = 1000
        height = im.shape[0]
        width = im.shape[1]
        #print '--------------------im.shape',im.shape
        mask_final = np.zeros((height,width))
        for i in range(lines):
            seg_id = seg_id + 1
            mask_path = os.path.join(dataset_root, 'FlickrLogos-v2','classes','masks')
            mask_path = os.path.join(mask_path,grand_path, img_id+'.mask.'+str(i)+'.png')
            if not os.path.exists(mask_path):
                print '-----------------------------mask_path not exist'
            #print '-----------------------------mask_path',mask_path
            mask = cv2.imread(mask_path)
            #print '------------------------------mask_shape,mask_final.shape',mask.shape,mask_final.shape
            mask_final[mask[:,:,0] == 255] = seg_id
        
        seg_label_name = os.path.join(seg_dir,'train_%s_%s.tif'%(grand,img_id))
        seg_path = os.path.join(dataset_root,seg_label_name)
        seg_image = Image.fromarray(mask_final)
        #seg_image = seg_image.convert('RGB')
        #seg_label_name = os.path.join(seg_dir,'train_%s.tif'%img_id)
        #seg_path = os.path.join(dataset_root,seg_label_name)
        #seg_image = Image.fromarray(seg)
        seg_image.save(seg_path)
        train_lst.write("%d\t%s\t%s\n"%(img_num, img_path, seg_path))
train_lst.close()
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
test_lst = open(os.path.join(dataset_root,lst_dir,'test.lst'),'w')
test_imgtxt_file = os.path.join(dataset_root, 'FlickrLogos-v2', 'testset-logosonly.relpaths.txt')
img_num = 0
lines = []
with open(test_imgtxt_file, 'r') as f:
    for line in f.readlines():
        if '\xef\xbb\xbf'  in line:
            str1 = line.replace('\xef\xbb\xbf','')
            lines.append(str1.strip())
        else:
            lines.append(line.strip())
    for line in lines:
        txtpath = str(line)
#with open(test_imgtxt_file, 'r') as f:
#    for txtpath in f:
        img_num = img_num + 1
        img_path = os.path.join(dataset_root, 'FlickrLogos-v2', txtpath)
        print '-----------',img_path
        test_lst.write("%d\t%s\n"%(img_num, img_path))
test_lst.close()



            




            
