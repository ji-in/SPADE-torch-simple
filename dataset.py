import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
from ast import literal_eval
import numpy as np
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, args, dataset_path):
        
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.channels = args.img_ch
        self.segmap_channel = args.segmap_ch
        self.augment_flag = args.augment_flag # Image augmentation use or not
        self.batch_size = args.batch_size

        self.dataset_path = dataset_path
        self.img_dataset_path = os.path.join(dataset_path, 'image')
        self.segmap_dataset_path = os.path.join(dataset_path, 'segmap')
        # self.segmap_test_dataset_path = os.path.join(dataset_path, 'segmap_test')

        self.image = glob(self.img_dataset_path + '/*.*')
        self.segmap = glob(self.segmap_dataset_path + '/*.*')
        self.color_value_dict = {}
        # self.segmap_test = []

    def preprocess(self): # segmap_label.txt가 없는 경우 만드는 함수
        # self.segmap_test = glob(self.segmap_test_dataset_path + '/*.*')
        segmap_label_path = os.path.join(self.dataset_path, 'segmap_label.txt')

        # semap_label.txt 가 존재하는 경우
        if os.path.exists(segmap_label_path) :
            print("segmap_label exists ! ")

            with open(segmap_label_path, 'r') as f:
                self.color_value_dict = literal_eval(f.read())
                print(self.color_value_dict)

        # semap_label.txt 가 존재하지 않는 경우
        # else :  
            # print("segmap_label no exists ! ")
            # label = 0
            # for img in tqdm(self.segmap) :

            #     if self.segmap_channel == 1 :
            #         # x = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE)
            #         x = Image.open(img) # 이미지를 읽어들인다.
            #     else : # self.semap_channel 이 1이 아닌 경우
            #         x = Image.open(img) # 이미지를 읽어들인다. Image는 이미 RGB

            #     x = x.resize((self.img_width, self.img_height))

            #     if self.segmap_channel == 1 :
            #         x = np.expand_dims(x, axis=-1) # channel이 1인 경우, 마지막에 차원을 추가한다.

            #     h, w, c = np.array(x).shape
                
            #     x = x.load()

            #     for i in range(h) :
            #         for j in range(w) :
            #             if tuple(x[i, j]) not in self.color_value_dict.keys() :
            #                 self.color_value_dict[tuple(x[i, j])] = label
            #                 label += 1
            #     # print(self.color_value_dict)
            # with open(segmap_label_path, 'w') as f :
            #     f.write(str(self.color_value_dict))


    # tensorflow는 (h, w, c), pytorch는 (ch, h, w)
    def convert_from_color_segmentation(self, color_value_dict, arr_3d): # arr_3d <- segmap (3, 256, 256)
        arr_2d = np.zeros((np.shape(arr_3d)[1], np.shape(arr_3d)[2]), dtype=np.uint8)
        
        for c, i in color_value_dict.items(): # (0, 255, 0): 15
            color_array = np.asarray(c, np.float32).reshape([1, 1, -1])
            m = np.all(arr_3d.numpy() == color_array, axis=0) # arr_ed == color_array 이면 True 아니면 False
            arr_2d[m] = i
        return torch.from_numpy(arr_2d)

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        data_transformer = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])

        image = Image.open(self.image[idx])
        image = data_transformer(image)

        segmap = Image.open(self.segmap[idx])
        segmap = data_transformer(segmap)

        label_map = self.convert_from_color_segmentation(self.color_value_dict, segmap) # (256, 256)
        segmap_onehot = F.one_hot(label_map.long(), len(self.color_value_dict)) # (h, w, c)로 나온다. 그래서 pytorch의 형태인 (c, h, w)로 바꿔야 한다.
        segmap_onehot = segmap_onehot.view(-1, label_map.shape[0], label_map.shape[1])
        segmap_onehot = segmap_onehot.float()

        return image, segmap, segmap_onehot

def imsave(input):
    input = input.numpy().transpose((1, 2, 0))
    plt.imshow(input)
    plt.savefig('test.png')

def load_dataset(args, dataset_path):
    
    train_datasets = CustomDataset(args=args, dataset_path=dataset_path)
    train_datasets.preprocess()
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # image, segmap, segmap_onehot = train_datasets[0]
    # imsave(segmap)
    # print(train_datasets.__len__())
    return train_dataloader

if __name__ == "__main__":

    dataset_path = './dataset/spade_celebA'
    img_dataset_path = os.path.join(dataset_path, 'image')
    segmap_dataset_path = os.path.join(dataset_path, 'segmap')
    image = glob(img_dataset_path + '/*.*')
    segmap = glob(segmap_dataset_path + '/*.*')

    # for i, (img, seg) in enumerate(zip(image, segmap)):
    #     print(img, seg)
    #     if i == 2:
    #         break

    for i, img in enumerate(image):
        print(img)
        if i == 3:
            break
    print("###############")
    for i, seg in enumerate(segmap):
        print(seg)
        if i == 3:
            break