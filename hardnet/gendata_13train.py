import cv2
import os
import numpy as np
import PIL.Image as Image
save_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966"
dataset_root = "/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/1030/Text-Detection/Text-Detection"
txt_dir = 'image_list'
seg_dir = 'seg_labels_13'
lst_dir = 'imglists_13'
#train_img_path = gb.glob("/dataset/user/6af39c4a-01d9-11e9-b0aa-fa163ee59f29/966/ch4_training_images/*.jpg")
#for path in train_img_path:
#   img = cv2.imread(path)
for filedir in [seg_dir,lst_dir]:
    filedir = os.path.join(save_root,filedir)
    if not os.path.exists(filedir):
        os.mkdir(filedir)
train_lst = open(os.path.join(save_root,lst_dir,'train.lst'),'w')
imgtxt_file = os.path.join(dataset_root, txt_dir, 'train_ic13.txt')
img_id = 0
with open(imgtxt_file, 'r') as f:
    for txtpath in f:
        img_id = img_id+1
        #if img_id == 392 or img_id == 379 or img_id == 383 or img_id == 389:
        #    continue
        if img_id > 400:
            break
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
        gt_path = os.path.join(dataset_root,gt_path)
        if not os.path.exists(gt_path):
            print '-----------------gt_path no exits',gt_path
            #continue
        with open(gt_path, 'r') as m:
            for line in m.readlines():
                if '\xef\xbb\xbf'  in line:
                    str1 = line.replace('\xef\xbb\xbf','')
                    lines.append(str1.strip())
                else:
                    lines.append(line.strip())
        im_size = cv2.imread(img_path).shape
        print '----------------------im_size',im_size
        #if len(im_size) != 3 or im_size[2]!=3:
        #    print '-----------------------not RGB',img_path
        #    continue 
        assert len(im_size)==3,"NOT RGB"
        assert im_size[2]==3,"NOT RGB"
        #print '-------------------',im_size[0], im_size[1]
        height = im_size[0]
        width = im_size[1]
        seg = np.zeros((height,width))
        seg_id = 1000
        for line in lines: 
            nums = line.split(',')
            if str(nums[8]) == "###":
                continue
            seg_id = seg_id+1
            A_x = int(nums[0])
            A_y = int(nums[1])
            B_x = int(nums[2])
            B_y = int(nums[3])
            C_x = int(nums[4])
            C_y = int(nums[5])
            D_x = int(nums[6])
            D_y = int(nums[7])
            x1 = int(min(A_x, B_x, C_x, D_x))
            x2 = int(max(A_x, B_x, C_x, D_x))
            y1 = int(min(A_y, B_y, C_y, D_y))
            y2 = int(max(A_y, B_y, C_y, D_y))
            #print '----------------------------',x1,x2,y1,y2
            if x1 <= 0:
                x1 = 1
            if y1 <= 0:
                y1 = 1
            if x2 > width-1:
                x2 = width-2
            if y2 > height-1:
                y2 = height-2
            for k in range(x1,x2):
                for j in range(y1,y2):
                    a = (B_x - A_x)*(j - A_y) - (B_y - A_y)*(k - A_x)
                    b = (C_x - B_x)*(j - B_y) - (C_y - B_y)*(k - B_x)
                    c = (D_x - C_x)*(j - C_y) - (D_y - C_y)*(k - C_x)
                    d = (A_x - D_x)*(j - D_y) - (A_y - D_y)*(k - D_x)
                    if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                        seg[j, k] = seg_id
        #seg_label_name = os.path.join(seg_dir,'train_%d.jpg'%img_id)
        #seg_path = os.path.join(save_root,seg_label_name)
        #seg_image = Image.fromarray(seg)
        #seg_image = seg_image.convert('RGB')
        seg_label_name = os.path.join(seg_dir,'train_%d.tif'%img_id)
        seg_path = os.path.join(save_root,seg_label_name)
        seg_image = Image.fromarray(seg)
        seg_image.save(seg_path)
        #img_dir = os.path.join(dataset_root,img_path)
        train_lst.write("%d\t%s\t%s\n"%(img_id-1, img_path, seg_path))
