import cv2
import os
from tqdm import tqdm
from glob import glob

dataset_path = './dataset/spade_celebA'
segmap_label_path = os.path.join(dataset_path, 'segmap_label.txt')

segmap_dataset_path = os.path.join(dataset_path, 'segmap')
segmap = glob(segmap_dataset_path + '/*.*')

img_width, img_height = 256, 256
color_value_dict = {}

if os.path.exists(segmap_label_path) :
    print("segmap_label exists ! ")

else :
    print("segmap_label no exists ! ")
    x_img_list = []
    label = 0
    for img in tqdm(segmap) :

        x = cv2.imread(img, flags=cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) # RGB로 고치기

        x = cv2.resize(x, dsize=(img_width, img_height), interpolation=cv2.INTER_NEAREST)

        h, w, c = x.shape

        x_img_list.append(x)

        for i in range(h) :
            for j in range(w) :
                if tuple(x[i, j, :]) not in color_value_dict.keys() :
                    color_value_dict[tuple(x[i, j, :])] = label
                    label += 1

    with open(segmap_label_path, 'w') as f :
        f.write(str(color_value_dict))