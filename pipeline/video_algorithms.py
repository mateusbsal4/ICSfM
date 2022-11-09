'''THIS CONTAINS KLT ALGORITHM'''

import cv2
import numpy as np
from os import path
import os

from pipeline import utils

debug_colors = np.random.randint(0, 255, (200, 3))


def klt_generator(config, file):
    """
    Generates and tracks KLT features from a video file.

    This function generates new features according to the parameters set in config and tracks
    old features as the video progresses. At each frame it tracks previous features
    and according to the number of frames to skip it yields a vector of features and their
    indexes.

    Feature indexes are unique and never repeating therefore can be used for indexing throughout
    the pipeline without collision.

    :param config: config object. See config.py for more information
    :param file: video file object
    :return: generator of tracks. Each item is of the form
        (
            frame_number: number of frame, monotonically increasing
            features: list of tuples with each feature location
            indexes: list of indexes for each returned features. These indexes
            are a global non repeating number corresponding to each feature
            and can be used to uniquely reference a reconstructed point
        )
    """

    reset_features = True

    prev_frame = None
    prev_features = np.empty((0, 2), dtype=np.float_)
    features = np.empty((0, 2), dtype=np.float_)           #creates empty arrays
    indexes = np.empty((0,))
    start_index = 100

    if config.calculate_every_frame:
        reset_period = config.reset_period * (config.frames_to_skip + 1)
        skip_at_getter = 0
        yield_period = config.frames_to_skip + 1
    else:
        reset_period = config.reset_period
        skip_at_getter = config.frames_to_skip
        yield_period = 1

    for frame_number, (color_frame, bw_frame) in enumerate(
        frame_getter(file, skip_at_getter)                      #yields single frames (rgb and grayscale) from video when called (will need changing!)
    ):
        if prev_frame is not None:
            features, indexes = track_features(
                bw_frame, config, indexes, prev_features, prev_frame
            )

        if reset_features or frame_number % reset_period == 0:      #all iterations will enter here (reset_features  ==1)
            reset_features = False

            features, indexes = get_new_features(
                bw_frame, config, features, indexes, start_index
            )
            start_index = max(indexes) + 1

        if frame_number % yield_period == 0:                    #here is where this function "returns", params will be yielded back to 
            yield frame_number, features, indexes               #the call on init_reconstruction (line 154)for each frame numbered "yield-period"

        prev_frame, prev_features = bw_frame, features

        if config.display_klt_debug_frames:
            display_klt_debug_frame(
                color_frame,                #yielded by frame_getter function (line 55)
                features,
                prev_features,
                indexes,
                config.klt_debug_frames_delay,
            )


def get_new_features(bw_frame, config, features, indexes, start_index):
    """
    Calculates new features to track and mergest it with previous ones

    :param bw_frame: frame in black and white
    :param config: config object. See config.py for more information
    :param features: previous features
    :param indexes: previous features indexes
    :param start_index: next index to be used on new features
    :return: new set of features to be tracked and corresponding indexes
    """

    new_features = cv2.goodFeaturesToTrack(                 #the function returns a 3D array e.g. [[[x1,y1],[x2,y2],...]] 
        image=bw_frame,                                     #and is then reshaped to an array of points: [[x1,y1],...]
        maxCorners=config.max_features,
        qualityLevel=config.corner_selection.quality_level,
        minDistance=config.corner_selection.min_distance,
        mask=None,
        blockSize=config.corner_selection.block_size,
    ).reshape((-1, 2))                                  

    features, indexes = match_features(
        features,
        indexes,
        new_features,
        start_index,
        config.closeness_threshold,
        config.max_features,
    )
    return features, indexes


def track_features(bw_frame, config, indexes, prev_features, prev_frame):       #doesn't need changing
    """
    Tracks a given set of indexes returning their new positions and dropping features
    that are not found

    :param bw_frame: frame in black and white
    :param config: config object. See config.py for more information
    :param indexes: previous features indexes
    :param prev_features: features from previous frames
    :param prev_frame: previous frame
    :return: new location of features and indexes
    """

    features, status, err = cv2.calcOpticalFlowPyrLK(           #calculates optical flow (vx,vy) and returns the feature positions in the next frame 
        prevImg=prev_frame,                                     #a status vector of 1s and 0s saying wether a feature has been found in the next frame or not   
        nextImg=bw_frame,                                       # and an err vector to calculate the error between actual and estimated positions (not applicabble)
        prevPts=prev_features,
        nextPts=None,
        winSize=(                                   #size of the window in which pixels are used for flow calculation 
            config.optical_flow.window_size.height,
            config.optical_flow.window_size.width,
        ),
        maxLevel=config.optical_flow.max_level,             
        criteria=(
            config.optical_flow.criteria.criteria_sum,
            config.optical_flow.criteria.max_iter,
            config.optical_flow.criteria.eps,
        ),
    )
    status = status.squeeze().astype(np.bool)       #removes unnecessary dimensions from status and sets its values(1 and 0) to true or false
    indexes = indexes[status].reshape((-1,))
    features = features.squeeze()[status]
    return features, indexes        #only indexes whose status is true are returned


def match_features(     # this doesn't need changing
    old_features,
    old_indexes,
    new_features,
    index_start,
    threshold,
    max_features,
):
    """
    Given a set of old features and new proposed features, it selects which features to use.

    Selection is done by calculating the euclidean distance between all features from
    old_features vector 

    :param old_features:
    :param old_indexes:
    :param new_features:
    :param index_start:
    :param threshold:
    :param max_features:
    :return:
    """
    if len(old_features) == 0:          #first iteration
        return new_features, np.arange(len(new_features)) + index_start

    closeness_table = get_close_points_table(
        new_features, old_features, threshold           
    )

    new_points_mask = closeness_table.sum(axis=0) == 0  #matrix is outside the screen so aixs 0 of the tensor corresponds to sum for each new feature compared to the old ones
                                                        #so an array of trues and falses is formed 
    new_features = new_features[new_points_mask]        #only new features far from all the others are selected
    new_indexes = np.arange(len(new_features)) + index_start #indexes are updated

    # limit number of returned features
    points_to_keep = min(max_features - len(new_features), len(old_features))       #this is necessary so that the number of detected features does not exceed param max_features
    old_features_mask = choose_old_features(closeness_table, points_to_keep)

    old_features = old_features[old_features_mask]
    old_indexes = old_indexes[old_features_mask]

    features = np.vstack((old_features, new_features))
    indexes = np.concatenate((old_indexes, new_indexes))

    assert len(features) == len(indexes)

    return features, indexes


def choose_old_features(closeness_table, points_to_keep):           #[ [1 1 1 1]
    """                                                             #  [1 1 0 0]...]
    Based on the number of close features (from closeness_table) it chooses which features
    to keep and which to discard

    :param closeness_table: table of 0s and 1s where a 1 indicates that the ith point
    from the old feature set is close enough to the jth point from the new feature set
    :param points_to_keep: number of points from old feature set to keep
    :return: mask of bools indicating which features to keep
    """

    mask = np.full(closeness_table.shape[0], False)     #creates a 1D-array of n (number of old features) falses
    indexes = np.empty([0], dtype=int)        #empty arrray                                                                  
    table_sum = closeness_table.sum(axis=1) #now sum is made on the "outer" arrays of the tensor -analysis on how similar each old feature is to the set of new ones

    base_indexes = np.arange(len(mask)) #(0,1,2,...,n)

    for sum_threshold in range(max(table_sum + 1)):  #analysing each old feature correspondence (given by table_sum) 
        points_to_go = points_to_keep - len(indexes)
        threshold_mask = table_sum == sum_threshold #array of length n [0,0,0,1,..] telling which ones of the old features are close to exactly "sum_threshold" new_features

        if sum(threshold_mask) <= points_to_go:    #enters here if not 
            indexes = np.hstack((indexes, base_indexes[threshold_mask]))    #concatenates both arrays -> resulting in addition of indexes of features which have proximity 
        else:
            indexes = np.hstack(
                (
                    indexes,
                    np.random.choice(
                        base_indexes[threshold_mask],
                        points_to_go,
                        replace=False,
                    ),
                )
            )

        assert len(indexes) <= points_to_keep

        if len(indexes) == points_to_keep:
            mask[indexes] = True
            break

    return mask

        
def get_close_points_table(new_features, old_features, threshold):
    """
    Generated the closeness table of 0s and 1s where a 1 indicates that the ith point
    from the old feature set is close enough to the jth point from the new feature set.
                                                                                                            
    The size of the table is len(old_features) lines and len(new_features) columns.

    :param new_features: list of new features to track
    :param old_features: list of old features to track
    :param threshold: distance (in pixels) below which two points are considered close      
    :return:                                        
    """
    old_repeated = np.repeat(                                   #repeat creates an array with multiple copies of each feature (so for n lines with features)                SHAPES IN NUMPY: |
        old_features, new_features.shape[0], axis=0             #in old_features, [[x1,x2]                  [[x1,x2]]                                                                        . __ (DOWN, OUT, TO THE SIDE)  
    ).reshape((old_features.shape[0], new_features.shape[0], 2))#                  [x3,x4]                   [x1,x2]      
    new_repeated = (                                            #                     ...           ->        ...           (m times, where m is the number
        np.repeat(new_features, old_features.shape[0], axis=0)  #                  [xn,xn+1]]                [x1,x2]            of new features)
        .reshape((new_features.shape[0], old_features.shape[0], 2))#                                         [x3,x4]         generates m xn matrix                
        .transpose((1, 0, 2))                                                                               #...
    )                                                               #the reshape will then turn this into a tensor of n(down)xm(out of the screen) where each out of the screen line contains m copies of a feature
    distances = np.linalg.norm(old_repeated - new_repeated, axis=2) #the new_repeated tensor of shape (mxnx2) is formed the same way but then transposed to (nxmx2)   
    close_points = distances < threshold                            #thennxm norm is calculated 
    return close_points
 

def frame_getter(file, frames_to_skip):
    """
    Gets a frame from file, skips frames_to_skip times and returns the color and
    grayscale version of next frame

    :param file: video file
    :param frames_to_skip: number of frames to skip between returned frames
    :return: original video frame and grayscale version
    """
    while True:
        for _ in range(frames_to_skip + 1):     #frames to skip won't be yielded
            ret, color_frame = file.read()
            if not ret:
                return
        bw_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        yield color_frame, bw_frame


def get_video(file_path):
    """
    Loads file from file_path

    :param file_path:
    :return:
    """
    # print("Looking for files in {}".format(dir))

    assert path.isfile(file_path), "Invalid file location"

    file = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)

    return file, filename


def display_klt_debug_frame(
    color_frame, features, prev_features, indexes, delay
):
    """
    Display a frame and colored dots indicating which features are being tracked

    :param color_frame: original frame
    :param features: list of features being tracked
    :param prev_features: list of previous feature positions
    :param indexes: list of featue indexes
    :param delay: wait time before exiting visualization
    :return: None
    """
    mask = np.zeros_like(color_frame)       #3 channel matrix filled with zeros (black image)

    for feature, prev_feature, index in zip(features, prev_features, indexes):      #creates a tuple of ((features[0],prev_features[0],indexes[0]),(features[1],prev_features[1],indexes[1]),...)
        # next_x, next_y = feature                                                  #where each feature and prev_feature is a point [x,y]
        # prev_x, prev_y = prev_feature

        mask = cv2.line(                                #lines will only appear if cofig.calculate_every_frame is false 
            mask,
            # (next_x, next_y),
            # (prev_x, prev_y),
            tuple(feature),
            tuple(prev_feature),
            debug_colors[index % 200].tolist(),             #debug_colors is a 200x3 array with numbers from 0 to 255, so 
            2,                                              #we're accessing (r,g,b) element of that list
        )
        color_frame = cv2.circle(
            color_frame,
            tuple(feature),
            5,
            debug_colors[index % 200].tolist(),
            -1,
        )

    img = cv2.add(color_frame, mask)                        #adds two images

    cv2.imshow("frame", img)
    cv2.waitKey(delay)
