from importlib.resources import path
import numpy as np
import glob
  
def parse_single_yolo_annot(path_to_annot):
    """
    parses a single yolo annotation text file 
    returns: frame number, and a list of 4-tuples (x, y, w, h)
    """

    with open(path_to_annot) as f:
        annotations = f.readlines()
    annotations = [i.strip() for i in annotations]                     # remove newlines
    annotations = [i.split(" ") for i in annotations]                  # convert strings to lists 
    annotations = [i[1:] for i in annotations]                         # remove classes
    bboxes = [[float(i) for i in bbox] for bbox in annotations]        # cast to float
    frame = int(path_to_annot.split(".")[0].split("_")[-1])            # get frame number 
    return frame, bboxes 


def parse_yolo_annot(filename, label_dir):
    """
    parses all annotations for a given video file into a dictionary
    """ 
    annot_files = glob.glob(f"{label_dir}/{filename}_*")               
    annotations = dict()                                               
    for path_to_annot in annot_files:                                  
        frame, bboxes = parse_single_yolo_annot(path_to_annot)
        annotations[frame] = bboxes

    return annotations

def parse_urbansas_annot(filename, label_dir):
    """
    parses urbansas annotations
    To remove after fixing the centre problem
    """
    annot_files = glob.glob(f"{label_dir}/{filename}_*")               
    annotations = dict()                                               
    for path_to_annot in annot_files:                                  
        frame, bboxes = parse_single_yolo_annot(path_to_annot)
        for i in range(len(bboxes)):
            bboxes[i][0] += bboxes[i][2]/2
            bboxes[i][1] += bboxes[i][3]/2
            
        annotations[frame] = bboxes

    return annotations



def remove_non_moving(annotations, iou_thresh):
    """
    removes non-moving bounding boxes 
    bboxes with an iou higher than iou_thresh in subsequent frames would be removed
    arguments:
            1. annotations (dict)
            2. iou_thresh (float) - a threshold for iou between bounding boxes
                                    values in [0,1]
    returns:
            1. annotions - filtered version of the annotations
    """

    frames = sorted(annotations.keys())
    for idx in range(len(frames) -1):
        bboxes1 = annotations[frames[idx]]
        bboxes2 = annotations[frames[idx+1]]

        iou_matrix = pairwise_iou(bboxes1, bboxes2)
        max_iou = np.max(iou_matrix, axis=1)
        to_keep = [max_iou<iou_thresh][0]
        bboxes1 = [box for idx, box in enumerate(bboxes1) if to_keep[idx]]
        annotations[frames[idx]] = bboxes1

        if idx == len(frames) - 2:
            max_iou = np.max(iou_matrix, axis=0)
            to_keep = [max_iou<iou_thresh][0]
            bboxes2 = [box for idx, box in enumerate(bboxes2) if to_keep[idx]]
            annotations[frames[idx+1]] = bboxes2


    return annotations

def bbox_iou(box1, box2, eps = 1e-9):
    """calculates iou (intersection over union) for two boxes (x,y,w,h)
    """
    # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # area of intersection 
    inter = max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1))) * \
            max(0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)))

    # area of union
    w1, h1 = box1[2], box1[3] 
    w2, h2 = box2[2], box2[3] 
    union = w1*h1 + w2*h2 - inter + eps

    # intersection over union
    iou = inter/union

    return iou

def pairwise_iou(boxes1, boxes2):
    """calculates pairwise iou between two lists of bounding boxes
    returns a matrix of iou values with dimensions - len(boxes1) x len(boxes2)
    """
    l1, l2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((l1, l2))

    for i in range(l1):
        for j in range(l2):
            iou_matrix[i, j] = bbox_iou(boxes1[i], boxes2[j])
    
    return iou_matrix
