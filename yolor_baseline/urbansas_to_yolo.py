"""
This script converts the urbansas data to the YOLO format 

Directory structure:

├── yolor_baseline
|   ├── yolor
|   ├── urbansas_filtered
│   |   ├── train
|   |   |   ├── images
|   |   |   └── labels 
│   |   ├── valid
|   |   ├── test
|   |   └── data.yaml

Conversions - 

videos -> images
video_annotations.csv -> yolo style lables

1. Do a train test split

2.  for train and test 
        for video in videos 
            a. get annotations for that video
            b. loop through all the frames of the video
                - get annotations for frame (if not present interpolate)
                - convert annotations to yolo
                - save image 
                - write label file

3. write data.yaml file 

"""

import pandas as pd 
import numpy as np
import cv2 as cv
import os 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

video_dir = "../data/urbansas_filtered/merged/"
annotation_path = "../data/urbansas_filtered/video_annotations.csv"
sampling_rate = 2                          # the rate at which to sample images from videos in fps

# setup directories 
dirs = ["urbansas_filtered/train/images",
        "urbansas_filtered/train/labels",
        "urbansas_filtered/valid/images",
        "urbansas_filtered/valid/labels"]
for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)

def main():

    annotations = pd.read_csv(annotation_path)
    # filter negative values for bbox coordinates
    annotations = annotations[(annotations["x"]>=0)&(annotations["w"]>=0)]
    annotations = annotations[(annotations["y"]>=0)&(annotations["h"]>=0)]

    filenames = np.array(annotations["filename"].unique())

    # train val split
    train_vids, val_vids = train_test_split(filenames, test_size = 0.2)
    print(f"train size = {len(train_vids)}\ntest size = {len(val_vids)}")

    train_test = {"train":train_vids,
                  "valid":val_vids}

    # for both train and test
    for phase in ["train", "valid"]:
        print(f"Creating the {phase} set...")
        vids = train_test[phase]
        for vid in tqdm(vids): 
            # load video
            filepath = os.path.join(video_dir, f"{vid}.mp4")
            cap=cv.VideoCapture(filepath)                     
            fps = int(cap.get(cv.CAP_PROP_FPS))
            nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cap_frame = 0

            # get annotations for the video
            sub_annot = annotations[annotations["filename"] == vid]
            sub_annot["frame_num"] = [int(i) for i in sub_annot["time"]*fps]

            # TODO: Interpolate annotations

            # normalise annotations with width and height
            sub_annot["x"]/=width
            sub_annot["w"]/=width
            sub_annot["y"]/=height
            sub_annot["h"]/=height

            # change x and y from top-left to centre
            sub_annot["x"]+=(sub_annot["w"]/2)
            sub_annot["y"]+=(sub_annot["h"]/2)


 
            for frame_num in sub_annot["frame_num"]:
                # get annotations for frame 
                frame_annot = sub_annot[sub_annot["frame_num"] == frame_num]

                # get image 
                while cap_frame < frame_num:
                    ret, img = cap.read()
                    cap_frame+=1
                ret, img = cap.read()
                if ret:
                    # save image 
                    img_name = f"{vid}_{frame_num}.jpg"
                    cv.imwrite(f"urbansas_filtered/{phase}/images/{img_name}", img)

                    # write label file
                    yolo_annot = create_annot_txt(frame_annot)
                    annot_name = f"{vid}_{frame_num}.txt"
                    with open (f"urbansas_filtered/{phase}/labels/{annot_name}", "w") as filehandler:
                        filehandler.writelines(yolo_annot)

    # write data.yaml file 
    with open ("urbansas_filtered/data.yaml", "w") as filehandler:
        data_yaml = ["train: ../urbansas_filtered/train/images\n",
                     "val: ../urbansas_filtered/valid/images\n",
                     "nc: 1\n",
                     "names: ['vehicles']\n"]

        filehandler.writelines(data_yaml)




    
def create_annot_txt(annotations):
    yolo_annot = []
    for i in range(annotations.shape[0]):
        row = annotations.iloc[i]
        yolo_annot.append(" ".join(["0",
                                    str(row["x"]),
                                    str(row["y"]),
                                    str(row["w"]),
                                    str(row["h"])]))
    yolo_annot = [i+"\n" for i in yolo_annot]
    return yolo_annot


if __name__ == "__main__":
    main()