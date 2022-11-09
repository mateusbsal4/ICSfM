import argparse
from cgitb import text
import itertools
from itertools import product
from pickletools import dis

import numpy as np
import time
import dacite
import cv2
from time import time
from termcolor import colored #[mbs:221101] module for printing in different colors
from operator import itemgetter
from ruamel.yaml import YAML
from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.reconstruction_algorithms import (
    calculate_projection,
    calculate_projection_error,
    solve_pnp,
)
from pipeline.utils import ErrorMetric
from pipeline.video_algorithms import get_video, klt_generator
from pipeline.init_algorithms import five_pt_init
from pipeline.config import VideoPipelineConfig


class VideoPipeline:
    """
    Main class used for reconstruction. It orchestrates the other components of the
    pipeline and interfaces with the user
    """

    def __init__(self, config: VideoPipelineConfig,) -> None:
        self.config = config

        self._first_reconstruction_frame_number = None

        # self.config.camera_matrix = np.array(self.config.camera_matrix)

    def run(self):
        """
        Given a set of config parameters passed at init this function runs and returns
        the reconstruction alongside with some error measurements
        :return: (
            Rs: list of rotation matrices from reconstructed cameras
            Ts: list of translation vectors from reconstructed cameras
            cloud: reconstruction point cloud
            init_errors: errors calculated during reconstruction init phase
            online_errors: errors calculated during reconstruction
            post_errors: errors calculated after reconstruction is finished
            execution_time: how many second it took to run the pipeline (wall time)
        """
        track_generator = self._setup(self.config.file_path)

        start = time()
        (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            track_generator,
            init_errors,
        ) = self._init_reconstruction(track_generator)
    
        if cloud is None:
            # init faild
            return None, None, None, init_errors, [], [], time() - start

        (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            online_errors,              #after init phase the algorithm also uses error_calculation_frames
            post_errors,
        ) = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks, masks, frame_numbers
        )
        execution_time = time() - start

        return (
            Rs,
            Ts,
            cloud,
            init_errors,
            online_errors,
            post_errors,
            execution_time,
        )

    def _setup(self, dir):
        """
        Instantiated frame generator

        This function is separated from run() so that child classes can modify it
        easily

        :param dir: filepath of video file
        :return: generator of raw frames
        """
        file, _ = get_video(dir)
        track_generator = klt_generator(self.config.klt, file)
        return track_generator

    def _init_reconstruction(self, track_generator):
        """
        Runs initial reconstruction phase for a given track generator

        It loades K + P frames, feed the first K frames to the reconstruction
        function for a initial reconstruction attempt then uses the other P
        frames for error calculation with out of sample frames.

        If calculated error is below threshold it adds back those frames used
        for error calculation back to generator and return the reconstruction result.

        If calculated error is above threshold it drops the first frame, load a
        new one and repeats everything.

        :param track_generator: generator of tracks. Each item is of the form
            (
                frame_number: number of frame, monotonically increasing
                features: list of tuples with each feature location
                indexes: list of indexes for each returned features. These indexes
                are a global non repeating number corresponding to each feature
                and can be used to uniquely reference a reconstructed point
            )

        :return: (
            Rs: list of rotation matrices from reconstructed cameras
            Ts: list of translation vectors from reconstructed cameras
            cloud: reconstruction point cloud
            init_errors: errors calculated during reconstruction init phase
            tracks: tracks used for reconstruction
            masks: masks used for reconstruction
            frame_numbers: indexes of frames used for reconstruction
            track_generator: same track generator as the one from the input but
            with reconstruction and dropped frames removed
            init_errors: errors calculated during init phase
        )
        """
        config = self.config.init

        tracks = []
        masks = []
        frame_numbers = []

        init_errors = []

        dropped_tracks = 0

        for frame_number, track_slice, mask in track_generator:
            tracks += [track_slice]
            masks += [mask]
            frame_numbers += [frame_number]

            #i = 0 ##[mbs: 221020]: tests

            if (
                len(tracks)
                < config.num_reconstruction_frames
                + config.num_error_calculation_frames   #[mbs:221007] commented for testing
            ):
                continue            #nothing is done until the 10th frame

            # update number of supposed first frame for reconstruction
            self._first_reconstruction_frame_number = frame_numbers[0]         #[mbs:221021] commented for testing

           # self._first_reconstruction_frame_number = frame_numbers[i]      #[mbs:221021] added for testing            

            reconstruction = self._reconstruct(
                zip(
                    frame_numbers[: config.num_reconstruction_frames],         #[mbs:221021] commented for testing
                    tracks[: config.num_reconstruction_frames],
                    masks[: config.num_reconstruction_frames],
                ),              #reconstruction called with the first 5 frames, lists of features and indexes-> (([1],[(90,310),(dc,wc),...],[1,2,3,4,...]),([2])
                is_init=True,
            )

            #reconstruction = self._reconstruct(
            #    zip(
            #        frame_numbers[i: config.num_reconstruction_frames+i],
            #        tracks[i: config.num_reconstruction_frames+i],               #[mbs:221021] added for testing
            #        masks[i: config.num_reconstruction_frames+i],
            #    ),              #reconstruction called with the first 5 frames, lists of features and indexes-> (([1],[(90,310),(dc,wc),...],[1,2,3,4,...]),([2])
            #    is_init=True,
            #)


            cloud = reconstruction[2]
            #print("Cloud", cloud)
            if cloud is None:
                tracks.pop(0)                                                             
                masks.pop(0)
                frame_numbers.pop(0)
                dropped_tracks += 1
                continue
            #if (cloud is None):
            #    i+=1
            #    continue            #[mbs: 221020]: will this do?
#



            # call error calculation with reconstruction and P frames

            error = self._calculate_init_error(
                tracks[-config.num_error_calculation_frames :],
                masks[-config.num_error_calculation_frames :],              #P=5 in this case
                frame_numbers[-config.num_error_calculation_frames :],      #reprojection error is calculated (and pnp is run) only with the first five-frame cloud
                cloud,                                                      #[mbs:221001] added for testing
            )                                                               #[mbs:221021] commented for testing


            #error = self._calculate_init_error(
            #    tracks[-config.num_error_calculation_frames +i:i],
            #    masks[-config.num_error_calculation_frames+i :i],             
            #    frame_numbers[-config.num_error_calculation_frames+i :i],      
            #    cloud,                                                      #[mbs:221021] added for testing
            #)                  
            
            #error = self._calculate_init_error(
            #    tracks[-config.num_error_calculation_frames :],            #[mbs:221001] commented for testing
            #    masks[-config.num_error_calculation_frames :],             
            #    [frame_numbers[0]],
            #    cloud,
            #)




            init_errors += [error] #[mbs:221007] commented for testing

            #print(
            #    f"{self.config.bundle_adjustment.use_at_end},"
            #    f"{self.config.bundle_adjustment.use_with_rolling_window},"
            #    f"{self.config.bundle_adjustment.rolling_window.method},"
            #    f"{self.config.synthetic_config.noise_covariance},"
            #    f"{config.num_reconstruction_frames},"
            #    f"{config.num_error_calculation_frames},"
            #    f"{dropped_tracks},"
#           #     f"{dropped_tracks}"       #[mbs:221007] added for testing            
            #    f"{error}"     #[mbs:221007] commented for testing
            #)
#
            # exit init or or drop first track/mask
            if error.projection > config.error_threshold:
                # drop first track ank mask and rerun the process
                tracks.pop(0)                                                              #[mbs:221007] commented for testing
                masks.pop(0)
                frame_numbers.pop(0)
                dropped_tracks += 1
            else:
                # add tracks used for error calculation back to track generator
                track_generator = itertools.chain(                                                 #[mbs:221007] commented for testing
                    zip(                                                                   
                        frame_numbers[-config.num_error_calculation_frames :],
                        tracks[-config.num_error_calculation_frames :],
                        masks[-config.num_error_calculation_frames :],
                    ),
                    track_generator,
                )
                return reconstruction[:6] + (track_generator, init_errors) #[mbs:221007] commented for testing
            
    #        return reconstruction[:6] + (track_generator, None) #[mbs:221007] added for testing
        else:
           return (None,) * 7 + (init_errors,)
        #    return (None,) * 7 + None           #[mbs:221007] added for testing
    
    def _reconstruct(
        self,
        track_generator,
        Rs=None,
        Ts=None,
        cloud=None,
        tracks=None,
        masks=None,
        frame_numbers=None,
        is_init=False,
    ):
        """
        This is the main reconstruction function, both for the init and
        increments reconstruction phases.

        If partial reconstruction (with the variables Rs, Ts, cloud, tracks,
        masks, frame_numbers not None and is_init=False) it continues the reconstruction,
        otherwise it begins one from scratch.

        :param track_generator: generator of features, indexes and frame number
        :param cloud: point cloud with N points as a ndarray with shape Nx3
        :param Rs: list of R matrices used for initial reconstruction
        :param Ts: list of T vectors used for initial reconstruction
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of frame indexes/numbers used for initial reconstruction
        :param is_init: flag inficating if it's in the initial reconstruction phase
        :return: (
            Rs: list of rotation matrix
            Ts: list of translation vectors
            cloud: list of reconstructed 3d points
            tracks: list of tracks
            masks: list of trac masks
            frame_numbers: list of frame numbers
            online_errors: list of errors calculated during reconstruction
            post_errors: list of errors calculated after reconstruction is finished
        )
        """

        if tracks is None:
            tracks, masks, frame_numbers = [], [], []

        # Before processing any frames, calculate error metrics for current frames (after initialization)
        if not is_init and self.config.error_calculation.online_calculation:            
            online_errors = self._calculate_reconstruction_errors_from_history(
                Rs, Ts, cloud, tracks, masks, frame_numbers
            )
        else:
            online_errors = []


        old_dis = 0 #[mbs:221029] local variable for first/second correction attempts
        last_peak = False    #[mbs:221030] second attempt
        if not is_init:
            print(colored("INIT IS OVER", 'green')) #[mbs:221102] debugging
            T_comp=Ts[-1]

        for frame_number, track_slice, index_mask in track_generator:           #this will run for the initial k+2 frames(config.init.num_reconstruction_frames)
            tracks += [track_slice]
            masks += [index_mask]
            frame_numbers += [frame_number]

            # Init cloud
            if cloud is None:
                Rs, Ts, cloud = five_pt_init(self.config, tracks, masks)            #until the 2nd frame
                continue
            assert(cloud is not None)       
            # Reconstruct
            R, T, points, index_mask = calculate_projection(
                self.config, tracks, masks, Rs[-1], Ts[-1], cloud
            )

            #[mbs:221027]HERE GOES AN INITIAL FIX ATTEMPT
            #dis = np.linalg.norm(T-Ts[-1], axis=0) - old_dis
            #if dis > 1.0:
            #    tracks.pop()
            #    masks.pop()
            #    frame_numbers.pop()
            #    old_dis = dis
            #    continue


            #[mbs:221030] SECOND ATTEMPT - LARGE DISPLACEMENT/PEAK DETECTOR
            #dis = np.linalg.norm(T-Ts[-1], axis=0)-old_dis
            #if dis > 6.5 and not last_peak: 
            #    tracks.pop()
            #    masks.pop()
            #    frame_numbers.pop()
            #    old_dis = dis
            #    last_peak = True
            #    print("peak detection")
            #    continue
            #elif dis < 1.0 and last_peak:
            #    tracks.pop()
            #    masks.pop()
            #    frame_numbers.pop()
            #    old_dis = dis 
            #    print("close to peak")
            #    continue
            #last_peak = False
            #old_dis = 0 

            #[mbs:221031] THIRD ATTEMPT: PEAK REMOVER  
            #if not is_init:
            #    dis = np.linalg.norm(T-T_comp, axis=0)
            #    #thresh = 4          #[mbs:221101] very sensitive to adjusting (and must be high)
            #    thresh = 7        #[mbs:221107] optifa
            #    #thresh = 2.5       #[mbs:221107] optimal for xadrez_cd (and elefante-curto!)
            #    #thresh = 7       #[mbs:221102] this is the optimal thresh for xadrez_cc
            #    #thresh = 6.5        #[mbs:221103] optimal for casaquasercirccont
            #    if dis>thresh:           
            #        tracks.pop()
            #        masks.pop()
            #        frame_numbers.pop()
            #        continue 
            #    T_comp = T
            
                
                

            # Add new points to cloud
            if points is not None:
                cloud = utils.add_points_to_cloud(cloud, points, index_mask)

            # Save camera pose

            Rs += [R]
            Ts += [T]
            #assert(np.linalg.norm(Ts[-1]-Ts[-2], axis =0)<1) [mbs221027: testing corections]
            print("DISTANCIA", np.linalg.norm(Ts[-1]-Ts[-2], axis =0)) #[mbs:221025]:debugging
            if np.linalg.norm(Ts[-1]-Ts[-2], axis =0)>7:
                text = colored("DISTANCE HIGHER THAN SEVEN", 'red')   #[mbs:221101]:debugging
                print(text)
            if np.linalg.norm(Ts[-1]-Ts[-2], axis =0)>2.5:
                text = colored("DISTANCE", 'red')   #[mbs:221101]:debugging
                print(text, np.linalg.norm(Ts[-1]-Ts[-2], axis =0))
            # Run optimizations
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks)

            # Calculate and store error metrics
            if is_init or not self.config.error_calculation.online_calculation:
                continue

            online_errors += [
                self._calculate_reconstruction_error(
                    *self._select_reconstruction_error_data(
                        Rs, Ts, tracks, masks, frame_numbers        #error is calculated reprojecting cloud only in last frame
                    ),
                    cloud,
                )
            ]
            #print(online_errors[-1])
            #if not is_init:
            #    utils.visualize(self.config.camera_matrix, Rs, Ts, cloud)   #[mbs221026]: online visualizing 


         #Optimize at end, but only if it's not in init phase
        if not is_init:
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks, True)      #[mbs221025]: commented for testing
            

        if not is_init and self.config.error_calculation.post_calculation:
            post_errors = self._calculate_reconstruction_errors_from_history(
                Rs, Ts, cloud, tracks, masks, frame_numbers
            )
        else:
            post_errors = []

        return (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            online_errors,
            post_errors,
        )

    def _calculate_init_error(self, tracks, masks, frame_numbers, cloud):
        """
        Calculates the projection error for a set of frames given some init
         conditions.

        It calculates the error by first calculating the expected rotation and
        translation followed by the resulting 2D projection of this pose and
        then comparing it with the original track slice.

        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :param cloud: actual point cloud
        :return: mean error
        """
        # Rs and Ts are only used in the synthetic pipeline

        errors, Rs, Ts, = [], [], []
       
        for frame_number, track, mask in zip(frame_numbers, tracks, masks):
            #print("HOW??", (frame_number, track, mask)) #[mbs:221001] 
            R, T = solve_pnp(self.config.solve_pnp, track, mask, cloud)
            Rs += [R]
            Ts += [T]
            #print(Rs)
            #print(Ts)                      #[mbs:221001]   
            #print("entrou")

        # error = calculate_projection_error(
        #     self.config.camera_matrix, Rs, Ts, cloud, tracks, masks, mean=True
        # )

        error = self._calculate_reconstruction_error(
            Rs, Ts, tracks, masks, frame_numbers, cloud
        )

        return error

    def _calculate_reconstruction_error(                    #error is calculated only with respect to each frame's features (cloud points are filtered prior to reprojection) 
        self, Rs, Ts, tracks, masks, frame_numbers, cloud
    ):
        """
        
        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :param cloud: point cloud with N points as a ndarray with shape Nx3
        :return:
        """

        projection_error = calculate_projection_error(
            self.config.camera_matrix, Rs, Ts, cloud, tracks, masks, mean=True
        )


        #print("ERRO:", projection_error) [mbs:221026] not analysing this now
        error = ErrorMetric(
            frame_numbers[-1], projection_error, np.nan, np.nan, np.nan
        )

        return error

    def _select_reconstruction_error_data(
        self, Rs, Ts, tracks, masks, frame_numbers
    ):
        """
        Selects data to be used on reconstruction error calculation. Returns
        slices of inputs.

        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :return: slices of inputs with items to be used for calculation
        """
        if len(Rs) % self.config.error_calculation.period != 0:
            return [] * 5

        error_window = self.config.error_calculation.window_length

        return_slices = (
            Rs[-error_window:],
            Ts[-error_window:],
            tracks[-error_window:],
            masks[-error_window:],
            frame_numbers[-error_window:],
        )

        return return_slices

    def _calculate_reconstruction_errors_from_history(
        self, Rs, Ts, cloud, tracks, masks, frame_numbers
    ):
        """
        Similar to calculate_reconstruction_error but it calculates errors
        after all reconstruction. Reprojects current cloud with Rs and Ts of each previous frame and calculates error with respect to their features

        :param Rs:
        :param Ts:
        :param cloud:
        :param tracks:
        :param masks:
        :param frame_numbers:
        :return:
        """
        # assert (
        #     len(Rs)
        #     == len(Ts)
        #     == len(tracks)
        #     == len(masks)
        #     == len(frame_numbers)
        # )

        errors = []
        for i in range(1, len(Rs) + 1):
            errors += [
                self._calculate_reconstruction_error(
                    *self._select_reconstruction_error_data(
                        Rs[:i], Ts[:i], tracks[:i], masks[:i], frame_numbers[:i]
                    ),
                    cloud,
                )
            ]

        return errors

    def _run_ba(self, Rs, Ts, cloud, tracks, masks, final_frame=False):
        """
        Decides if the Bundle Adjustment step must be run and with how much data.

        :param Rs:
        :param Ts:
        :param cloud:
        :param tracks:
        :param masks:
        :param final_frame:
        :return:
        """

        config = self.config.bundle_adjustment

        if config.use_at_end and final_frame:
            Rs, Ts, cloud = bundle_adjustment.run(
                config, Rs, Ts, cloud, tracks, masks
            )

        elif (
            config.use_with_rolling_window
            and len(Rs) % config.rolling_window.period == 0
        ):
            method = config.rolling_window.method
            length = config.rolling_window.length
            step = config.rolling_window.step

            if method == "constant_step":
                ba_window_step = step
                ba_window_start = -(length - 1) * step - 1

                (
                    Rs[ba_window_start::ba_window_step],
                    Ts[ba_window_start::ba_window_step],
                    cloud,
                ) = bundle_adjustment.run(
                    config,
                    Rs[ba_window_start::ba_window_step],
                    Ts[ba_window_start::ba_window_step],
                    cloud,
                    tracks[ba_window_start::ba_window_step],
                    masks[ba_window_start::ba_window_step],
                )
            elif method == "growing_step":
                indexes = [
                    item
                    for item in [
                        -int(i * (i + 1) / 2 + 1)
                        for i in range(length - 1, -1, -1)
                    ]
                    if -item <= len(Rs)
                ]

                R_opt, T_opt, cloud = bundle_adjustment.run(
                    config,
                    itemgetter(*indexes)(Rs),
                    itemgetter(*indexes)(Ts),
                    cloud,
                    itemgetter(*indexes)(tracks),
                    itemgetter(*indexes)(masks),
                )

                for index, R, T in zip(indexes, R_opt, T_opt):
                    Rs[index] = R
                    Ts[index] = T

        return Rs, Ts, cloud
