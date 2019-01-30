import json, os, sys, time, pickle, h5py, cv2, numpy as np
sys.path.append('/usr/local/lib/')
import pyrealsense2 as rs
import multiprocessing as mp
from MoseqMulti_utils import *

def delete_bagfiles(working_directory, session_name):
    for f in os.listdir(working_directory+'/data/'):
        if session_name in f and '.bag' in f:
            os.system('rm '+working_directory+'/data/'+f)


def get_extractable_sessions(working_directory):
    metadata_files = [f for f in os.listdir(working_directory+'/data/') if 'metadata.json' in f]
    hdf5_files = [f for f in os.listdir(working_directory+'/data/') if '.hdf5' in f]
    print('Found',len(metadata_files),'metadata files')
    
    metadata_files = [f for f in metadata_files if not (f.split('_metadata')[0]+'.h5py' in hdf5_files)]
    return [f.split('_metadata')[0] for f in metadata_files]
        





def get_pipeline_from_file(filepath, metadata):
    config = rs.config()
    align = rs.align(rs.stream.color)
    rs.config.enable_device_from_file(config, filepath)
    pipeline = rs.pipeline()
    config.enable_stream(rs.stream.depth, metadata['parameters']['frame_width'],  metadata['parameters']['frame_height'], rs.format.z16, metadata['parameters']['fps'])
    config.enable_stream(rs.stream.color, metadata['parameters']['frame_width'],  metadata['parameters']['frame_height'], rs.format.bgr8, metadata['parameters']['fps'])
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = rs.playback(device)
    playback.set_real_time(False)
    return pipeline, align


import matplotlib.pyplot as plt
def extract_to_avi(PARAMS, session_name):
    
    working_directory = PARAMS['working_directory']
    metadata = json.load(open(working_directory+'/data/'+session_name+'_metadata.json'))
    height = metadata['parameters']['frame_height']
    width = metadata['parameters']['frame_width']
    fps = metadata['parameters']['fps']    
    
    def extract_one_camera(sn,queue):  
        
        path_prefix = working_directory+'/data/'+session_name+'_'+sn
        pipeline, align = get_pipeline_from_file(path_prefix+'.bag', metadata)

        color_video_pipe = None
        depth_video_pipe = None
        timestamps = []
        total_frames = 0
        
        try:
            while total_frames < metadata['num_frames']:
                if total_frames % 100==0: print('Cam',sn,':',total_frames,'/',metadata['num_frames'])
                frames = pipeline.wait_for_frames(4000)
                aligned_frames = align.process(frames)
                color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data())
                depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())

                depth_video_pipe = write_depth_frames(path_prefix+'_depth.avi', 
                                                depth_frame[None,:,:], 
                                                pipe=depth_video_pipe, 
                                                close_pipe=False, fps=fps)

                color_video_pipe = write_color_frames(path_prefix+'_color.mp4', 
                                                color_frame[None,:,:,::-1], 
                                                pipe=color_video_pipe, 
                                                close_pipe=False, fps=fps,
                                                pixel_format='rgb24', 
                                                codec='h264',)

                timestamps.append(rs.frame.get_timestamp(frames))
                total_frames += 1
        finally:
             print('STOPPED',sn,'at',total_frames)
             np.save(path_prefix+'_timestamps.npy', timestamps)
             color_video_pipe.stdin.close()
             depth_video_pipe.stdin.close()
             queue.put(sn)
             time.sleep(0.1)
    
    queue = mp.Queue(); finished_processes = set([])
    processes = {sn:mp.Process(target=extract_one_camera, args=(sn,queue)) for sn in metadata['serial_numbers']}
    for sn,p in processes.items(): p.start()
    while len(finished_processes) < len(processes):
        time.sleep(0.1)
        if not queue.empty():
            sn = queue.get()
            processes[sn].terminate()
            processes[sn].join()
            finished_processes.add(sn)               
            
    

