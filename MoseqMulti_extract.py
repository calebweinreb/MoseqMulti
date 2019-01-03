import json, os, sys, time, pickle, h5py, cv2, numpy as np
sys.path.append('/usr/local/lib/')
import pyrealsense2 as rs

def delete_bagfiles(working_directory, session_name):
    for f in os.listdir(working_directory+'/data/'):
        if session_name in f and '.bag' in f:
            os.system('rm '+working_directory+'/data/'+f)


def extract(PARAMS, session_name): 
    working_directory = PARAMS['working_directory']
    metadata = json.load(open(working_directory+'/data/'+session_name+'_metadata.json'))
    h5_path = working_directory+'/data/'+session_name+'.hdf5'
    outfile = h5py.File(h5_path,'w')
    for sn in metadata['serial_numbers']:
        cam_group = outfile.create_group(sn)
        cam_group.create_dataset('color', (metadata['num_frames'],metadata['parameters']['frame_height'],metadata['parameters']['frame_width'],3), dtype='uint8')
        cam_group.create_dataset('depth', (metadata['num_frames'],metadata['parameters']['frame_height'],metadata['parameters']['frame_width']), dtype='uint16')
        cam_group.create_dataset('timestamp', (metadata['num_frames'],), dtype='f')
    
        config = rs.config()
        align = rs.align(rs.stream.color)
        rs.config.enable_device_from_file(config, working_directory+'/data/'+session_name+'_'+sn+'.bag')
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth, metadata['parameters']['frame_width'],  metadata['parameters']['frame_height'], rs.format.z16, metadata['parameters']['fps'])
        config.enable_stream(rs.stream.color, metadata['parameters']['frame_width'],  metadata['parameters']['frame_height'], rs.format.bgr8, metadata['parameters']['fps'])

        pipeline.start(config)
        total_frames = 0
        try:
            while total_frames < metadata['num_frames']:
                if total_frames % 100==0: print('Cam',sn,':',total_frames,'/',metadata['num_frames'])
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color_frame = np.asanyarray(frames.get_color_frame().get_data())
                depth_frame = np.asanyarray(frames.get_depth_frame().get_data())  
                cam_group['color'][total_frames,:,:,:] = color_frame
                cam_group['depth'][total_frames,:,:] = depth_frame
                cam_group['timestamp'][total_frames] = rs.frame.get_timestamp(frames)
                total_frames += 1
        finally: pass

    aligned_frames, timestamps = get_aligned_frames(outfile, metadata, PARAMS)
    alignment = outfile.create_group('frame_alignment')
    alignment.create_dataset('timestamps',data=timestamps)
    alignment.create_dataset('frames',data=aligned_frames)
    alignment.create_dataset('serial_numbers', data=[sn.encode('ascii','ignore') for sn in metadata['serial_numbers']])
    outfile.close()
    
                
            
def get_extractable_sessions(working_directory):
    metadata_files = [f for f in os.listdir(working_directory+'/data/') if 'metadata.json' in f]
    hdf5_files = [f for f in os.listdir(working_directory+'/data/') if '.hdf5' in f]
    print('Found',len(metadata_files),'metadata files')
    
    metadata_files = [f for f in metadata_files if not (f.split('_metadata')[0]+'.h5py' in hdf5_files)]
    return [f.split('_metadata')[0] for f in metadata_files]
        

def get_aligned_frames(data, metadata, PARAMS):
    Ts = []
    for sn in metadata['serial_numbers']:
        T = data[sn]['timestamp'][:] - metadata['timestamps'][sn][10]
        Ts.append(T)

    aligned_frames = []
    timestamps = []
    for i,t in enumerate(Ts[0]):
        frame = [i]
        for T in Ts[1:]:
            frame.append(np.argmin(np.abs(T - t)))  
        times = [Ts[ii][frame[ii]] for ii in range(len( metadata['serial_numbers']))]
        if (np.max(times)-np.min(times)) < PARAMS['aligned_frame_windowsize']:
            aligned_frames.append(frame)
            timestamps.append(t)
    aligned_frames = np.array(aligned_frames)       
    print('Aligned',aligned_frames.shape[0],'out of',len(Ts[0]),'frames')
    return aligned_frames, timestamps
