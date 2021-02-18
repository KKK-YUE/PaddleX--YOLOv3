# PaddleX--YOLOv3
基于PaddleX的10分钟快速上手，对基元进行YOLOv3目标检测。

# PaddleX--YOLOv3
基于PaddleX的10分钟快速上手，对基元进行YOLOv3目标检测。
!pip install paddlex -i https://mirror.baidu.com/pypi/simple

#准备基元目标检测数据集
import zipfile
zip_file = zipfile.ZipFile('jiyuan.zip')
zip_list = zip_file.namelist()
print(zip_list)
for f in zip_list:
    zip_file.extract(f, path = 'jiyuan/') # 循环解压文件到指定目录
 
zip_file.close()

!paddlex --split_dataset --format ImageNet --dataset_dir jiyuan --val_value 0.5 --test_value 0.0

#模型训练

#配置GPU
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

#定义图像处理流程transforms
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
     transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224), 
    transforms.Normalize()
])

#定义数据集Dataset
train_dataset = pdx.datasets.ImageNet(
    data_dir='jiyuan',
    file_list='jiyuan/train_list.txt',
    label_list='jiyuan/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='jiyuan',
    file_list='jiyuan/val_list.txt',
    label_list='jiyuan/labels.txt',
    transforms=eval_transforms)

#模型开始训练
model = pdx.cls.MobileNetV3_small_ssld(num_classes=len(train_dataset.labels))
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.025,
    save_dir='output/mobilenetv3_small_ssld',
    use_vdl=True)
