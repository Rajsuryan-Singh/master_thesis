import sys
import pandas as pd
import cv2 as cv
import numpy as np
from ffpyplayer.player import MediaPlayer
from tkinter.filedialog import askopenfilename
# get video and audio annotations
VID_ANNOT = pd.read_csv("data/urbansas/video_annotations.csv").drop(columns="Unnamed: 0")
AUD_ANNOT = pd.read_csv("data/urbansas/audio_annotations.csv").drop(columns="Unnamed: 0")

def main(arg):

    # get a list of all videos
    videos = sorted(list(set(VID_ANNOT["filename"])))
     
    if len(arg) == 1:
        # select file
        filepath = askopenfilename()
        show_vid_wlabels(filepath)
    elif len(arg) == 2:
        filename = arg[1]
        filepath = f"data/urbansas/merged/{filename}.mp4"
        show_vid_wlabels(filepath)
    elif len(arg) == 3 and arg[1] == "all":
        # loop through all videos 
        start = int(arg[2])
        for i, filepath in enumerate(videos):
            if i >= start:
                filepath = f"data/urbansas/merged/{filepath}.mp4"
                show_vid_wlabels(filepath)
            with open("last_vid_idx.txt", "w") as f:
                f.write(str(i))
    else:
        print("Unexpected arguments. You can either pass a filename (without extension), or ('all', start) where start is the index of video you wish to start from")

        

def show_vid_wlabels(filepath):
    #filepath = "data/urbansas/merged/street_traffic-lyon-1110-41520.mp4"
    filename = filepath.split("/")[-1][:-4]           # without extension (.mp4)
    print(filename)
    # filter annotations by filename
    vid_annot = VID_ANNOT[VID_ANNOT["filename"] == filename]
    aud_annot = AUD_ANNOT[AUD_ANNOT["filename"] == filename]
  
    # load video
    cap=cv.VideoCapture(filepath)                     

    # get fps to set wait_time
    fps = int(cap.get(cv.CAP_PROP_FPS))
    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    vid_length = int(round(nframes/fps))
    wait_time = 1000//fps

    # get framewise bounding boxes
    bbox_dict = {i+1:[] for i in range(vid_length*2 +1)}
    for i, row in vid_annot.iterrows():
        if row["frame_id"] != -1:
            bbox_dict[row["frame_id"]].append([row["x"], row["y"], row["w"], row["h"], row["label"]])

    # get framewise audio labels
    frame_labels = [[] for frame in range(nframes)]                # initialise framewise audio labels
    for i, row in aud_annot.iterrows():
        label, start, end = row["label"], row["start"], row["end"]
        start_f = int(round(start*fps))
        end_f = int(round(end*fps))
        for idx in range(start_f, end_f):
            frame_labels[idx].append(label)
                         

    # stream video 
    frame_id = 1                                        # new frame_id every 0.5 seconds
    frame = 0
    next_update = fps*0.5

    while cap.isOpened():
        ret, img = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("End of video. Exiting ...")
            break
        
        draw_bbox(img, bbox_dict[frame_id])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, ", ".join(frame_labels[frame]),(30,50), font, 1,(255,0,0),2,cv.LINE_AA)
        cv.imshow('frame', img)

        if frame == 0:
            # start streaming audio
            player = MediaPlayer(filepath) 

        frame+=1
        if frame > next_update:
            frame_id+=1
            next_update+=fps*0.5

        if cv.waitKey(wait_time) == ord('q'):
            break

        
    cap.release()
    cv.destroyAllWindows()

# define bbox colors for different classes
label_to_color = {"car":(100, 255, 100), 
                  "bus":(255, 100, 100), 
                  "motorbike":(100, 100, 255),
                  "truck":(255, 255, 100)
                  }

def draw_bbox(img, bboxes, color = (100, 255, 100)):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    bboxes: array off bboxes in (x,y,w,h) format
    """

    for box in bboxes:
        x, y, w, h, label = box
        left = int(x)
        right = int(left + w)
        top = int(y)
        bottom = int(top + h)

        color = label_to_color[label]
        thickness = 5
        cv.rectangle(img,(left, top), (right, bottom), color, thickness)
        #cv.putText(img, label, (left, top - 12), color, thick//3)
    return img


if __name__ == "__main__":
    main(sys.argv)