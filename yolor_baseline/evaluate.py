from utils import *
import glob
import numpy as np
from tqdm import tqdm

inf_dir = "inference/valid"
gt_dir = "urbansas_filtered/valid/labels"
iou_thresh = 0.95

def main():
    # make a list of all the files
    files = glob.glob(f"{inf_dir}/*txt")
    iou_list = []
    # loop through all the files
    for filepath in tqdm(files):
        filename = "_".join(filepath.split("/")[-1].split("_")[:-1])
        # parse yolo annotations
        annotations = parse_yolo_annot(filename, inf_dir)
        # clean the annotations (remove non moving bounding boxes)
        annotations = remove_non_moving(annotations, iou_thresh)
        # load the ground truth 
        gt = parse_urbansas_annot(filename, gt_dir)
        # compare with ground truth (calculate IoU)
        iou_list += calc_iou(annotations, gt)
    
    iou_list = np.array(iou_list)
    mean_iou = np.mean(iou_list[iou_list>0])
    print(f"Mean IoU: {mean_iou}")
    

def calc_iou(annot1, annot2):
    """
    calculates the ious between two sets of annotations
    best match (by iou) is found for each box in annot1 
    returns a list of ious between boxes in annot1 and the best match
    """
    frames = annot1.keys()
    ious = []
    for frame in frames:
        bboxes1 = annot1[frame]
        bboxes2 = annot2[frame]
        iou_matrix = pairwise_iou(bboxes1, bboxes2)
        max_iou = list(np.max(iou_matrix, axis = 1))
        ious += max_iou

    return ious


if __name__ == "__main__":
    main()