import os
import time
import shutil
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg
def iou(boxes1, boxes2):

    #boxes1 = np.array(boxes1)
    #boxes2 = np.array(boxes2)
    #a=boxes1[: , 0]
    #b=boxes1[:, 1]
    boxes1_area = boxes1[0]  * boxes1[1]
    boxes2_area = boxes2[:, 0]  * boxes2[:, 1]

    left_up       = np.minimum(boxes1[0], boxes2[:,0])
    right_down    = np.minimum(boxes1[1], boxes2[:,1])

    #inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = left_up * right_down
    union_area    = boxes2_area + boxes1_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious
def kmeans(boxes, k, dist=np.median,seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances     = np.empty((rows, k)) ## N row x N cluster
    last_clusters = np.zeros((rows,))
    np.random.seed(seed)
    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    aveIOU=0.0
    while True:
        # 为每个点指定聚类的类别（如果这个点距离某类别最近，那么就指定它是这个类别)
        for icluster in range(k):
            distances[:,icluster] = 1 - iou(clusters[icluster], boxes)
        nearest_clusters = np.argmin(distances, axis=1)

        for i  in range(rows ):
            aveIOU=aveIOU+1-distances[i,nearest_clusters[i]]
        aveIOU=aveIOU/rows

	# 如果聚类簇的中心位置基本不变了，那么迭代终止。
        if (last_clusters == nearest_clusters).all():
            break
        # 重新计算每个聚类簇的平均中心位置，并它作为聚类中心点
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters,nearest_clusters,distances,aveIOU
def parse_annotation(annotation):

    line = annotation.split()
    bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
    #image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
    return bboxes
def parse_annotation(annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preporcess(np.copy(image), [416, 416], np.copy(bboxes))
        return image, bboxes
with open("/home/mpiuser/tensoflow/YOLOV3/data/dataset/mydata_train.txt", 'r') as f:
    txt = f.readlines()
    annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    np.random.shuffle(annotations)
boxes=np.array([557,480])

for annotation in annotations:
    image, bboxes  = parse_annotation(annotation)
    #bboxes = parse_annotation(annotation)
    for bbox in bboxes:
        w=bbox[2]-bbox[0]
        h=bbox[3]-bbox[1]
        box=np.array([w,h])
        boxes=np.vstack((boxes,box))
    
#=
for i in range(16):
    if i==0:
        continue
    else:
        clusters,nearest_clusters,distances,aveIOU = kmeans(boxes,i)
#print(clusters)
#print(clusters[:,0]*clusters[:,1])
    print(aveIOU)
print("aaa")



    