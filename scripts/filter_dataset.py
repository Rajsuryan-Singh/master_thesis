import pandas as pd
import os 
import sys
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm

# get video and audio annotations
VID_ANNOT_PATH = "data/urbansas/video_annotations.csv"
AUD_ANNOT_PATH = "data/urbansas/audio_annotations.csv"
VID_DIR = "data/urbansas/merged/"
FILT_VID_ANNOT_PATH = "data/urbansas_filtered/video_annotations.csv"
FILT_AUD_ANNOT_PATH = "data/urbansas_filtered/audio_annotations.csv"
FILT_VID_DIR = "data/urbansas_filtered/merged/"
max_vehicles = 2
min_dur = 4

if not os.path.isdir(FILT_VID_DIR):
    os.makedirs(FILT_VID_DIR)

def main():
    # load annotations
    vid_annot = pd.read_csv(VID_ANNOT_PATH).drop(columns="Unnamed: 0")
    aud_annot = pd.read_csv(AUD_ANNOT_PATH).drop(columns="Unnamed: 0")

    # initialise filtered annotations
    vid_annnot_filt = pd.DataFrame(columns=vid_annot.columns)
    aud_annnot_filt = pd.DataFrame(columns=aud_annot.columns)

    # get all filenames
    files = vid_annot.filename.unique()

    # get framewise vehicle count 
    vid_annot["count"] = np.zeros((vid_annot.shape[0], 1))
    framewise_cts = vid_annot[["filename","frame_id", "count"]].groupby(["filename","frame_id"]).count()

    # initialise segments 
    segments_all = dict()

    # loop through all the files
    for filename in tqdm(files):
        # framewise counts for a given file
        file_cts = framewise_cts.loc[filename]

        # get segments 
        segments = get_segments(file_cts,
                                max_vehicles=max_vehicles, 
                                min_dur = min_dur)

        # split and filter videos and annotations
        if segments:
            vid = mpy.VideoFileClip(f"{VID_DIR}{filename}.mp4")
            for i, seg in enumerate(segments):
                clip = vid.subclip((seg[0]-1)*0.5, (seg[1]-1)*0.5)       # subtract 1 as frame ids start from 1
                clip.write_videofile(f"{FILT_VID_DIR}{filename}_{i}.mp4", verbose=False, logger=None)

                # get annotations 
                vid_annot_seg, aud_annot_seg = get_segment_annotations(seg, filename, vid_annot, aud_annot)

                # modify filename column for vidverbose=False, logger=Noneeo and audio annotations
                vid_annot_seg["filename"] = [f"{filename}_{i}"]*vid_annot_seg.shape[0]
                aud_annot_seg["filename"] = [f"{filename}_{i}"]*aud_annot_seg.shape[0]

                # accumulate annotations
                vid_annnot_filt = vid_annnot_filt.append(vid_annot_seg)
                aud_annnot_filt = aud_annnot_filt.append(aud_annot_seg)


    # save annotations
    vid_annnot_filt.to_csv(FILT_VID_ANNOT_PATH)
    aud_annnot_filt.to_csv(FILT_AUD_ANNOT_PATH)

    print(f"The dataset has been filtered to have a maximum of {max_vehicles} vehicles in each frame. The minimum allowed duration is {min_dur}s.")

                


def get_segments(file_cts, max_vehicles = 2, min_dur = 3):
    """
    Get segments with a maximum of max_vehicles vehicles 

    Arguments:
        1. file_cts (pd.DataFrame): framewise vehicle counts
        2. max_vehicles (int): the maximum number of vehicles allowed in any frame
        3. min_dur (float): the minimum duration for a segment in seconds
    
    Returns:
        1. segments (list[tuple(int, int)]): indexes for starting and ending points of segments (in frames)
    """
    segments = []                                              # list of all segments in a given file
    
    # initialize pointers
    frames = file_cts.index.values
    curr = 0                                            
    start = 0                                           
    end = 0

    # total number of frames
    n_frames = len(frames)

    # loop through frames accumulating segments 
    while curr < n_frames:
        if file_cts.loc[frames[curr]].values[0] > max_vehicles:
            # if more vehicles than max_vehicles, move on to the next frame
            curr+=1
            start+=1
            end+=1

        else:        
            next = curr+1
            if next==n_frames:
                # add segment to list of segments if long enough
                if (frames[end] - frames[start])/2 > min_dur:
                    segments.append([frames[start], frames[end]])

                # reset segment
                curr+=1
                start+=1
                end=start

            elif file_cts.loc[frames[next]].values[0] > max_vehicles:
                # add segment to list of segments if long enough
                if (frames[end] - frames[start])/2 > min_dur:
                    segments.append([frames[start], frames[end]])
                
                # reset segment
                curr+=1
                start+=1
                end=start
            else:
                curr+=1
                end+=1
                
    return segments


def get_segment_annotations(segment, filename, vid_annot, aud_annot):

    # filewise annotation
    vid_annot_f = vid_annot[vid_annot["filename"] == filename]
    aud_annot_f = aud_annot[aud_annot["filename"] == filename]

    # get annotations for segment
    start, end = segment
    start_time, end_time = (start-1)*0.5, (end - 1)*0.5

    # video annotations
    vid_annot_seg = vid_annot_f[(vid_annot_f.frame_id >= start) & (vid_annot_f.frame_id <= end)]
    vid_annot_seg["time"]-=start_time                                   # set the start time as 0                           
    vid_annot_seg["frame_id"]-=(start + 1)                              # set the first frame id as 1

    # audio annotations
    # filter annotations that overlap with the segment
    aud_annot_seg = aud_annot_f[(aud_annot_f.end >= start_time) & (aud_annot_f.start <= end_time)]

    # modify the start and end times of annotations to stay within the segments
    aud_annot_seg["start"] = [max(i, start_time) for i in aud_annot_seg["start"]]    
    aud_annot_seg["end"] = [min(i, end_time) for i in aud_annot_seg["end"]]

    return vid_annot_seg, aud_annot_seg

    
if __name__ == "__main__":
    main()
