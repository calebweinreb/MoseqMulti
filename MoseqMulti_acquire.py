import json, os, sys, time, pickle, h5py, cv2, numpy as np
sys.path.append('/usr/local/lib/')
import pyrealsense2 as rs

def start_pipes(pipelines,configs,serial_numbers):
	profiles = {}
	for n in serial_numbers[::-1]:
		profile = pipelines[n].start(configs[n])
		profiles[n] = profile
	return profiles

def stop_pipes(pipelines):
	for n in pipelines:
		pipelines[n].stop()   
        
def get_metadata(serial_numbers,start_time,stop_time,num_frames,intrinsics, timestamps, PARAMS):
	metadata = {'parameters':PARAMS,
                'serial_numbers':serial_numbers,
                'start_time':start_time,
                'stop_time':stop_time,
                'num_frames':num_frames,
		        'intrinsics':intrinsics,
                'timestamps': timestamps}
	return metadata

def get_pipelines(serial_numbers, PARAMS):
	pipelines = {}
	configs = {}
	for n in serial_numbers:
		try:
			pipeline = rs.pipeline()
			config = rs.config()
			config.enable_device(n)
			config.enable_stream(rs.stream.depth, PARAMS['frame_width'], PARAMS['frame_height'], rs.format.z16, PARAMS['fps'])
			config.enable_stream(rs.stream.color, PARAMS['frame_width'], PARAMS['frame_height'], rs.format.bgr8, PARAMS['fps'])
			config.enable_record_to_file(PARAMS['working_directory'] + '/data/' + PARAMS['session_name']+'_'+n+'.bag')
			pipelines[n] = pipeline
			configs[n] = config
		except: 
			print('Error connecting to camera '+n)
	return pipelines, configs

def get_connected_devices():
	ctx = rs.context()
	ds5_dev = rs.device()
	devices = ctx.query_devices();
	serial_numbers = []
	for d in devices:
		print('Found device ',d)
		serial_numbers.append(str(d).split('S/N: ')[1].split(')')[0])
	if len(devices)==0:
		print('No devices found')
	return serial_numbers

def get_intrinsics(pipelines):
	intrinsics = {k:{} for k in pipelines.keys()}
	for n in pipelines:
		ins = pipelines[n].wait_for_frames().get_depth_frame().profile.as_video_stream_profile().intrinsics
		intrinsics[n]['ppx'] = ins.ppx
		intrinsics[n]['ppy'] = ins.ppy
		intrinsics[n]['fx'] = ins.fx
		intrinsics[n]['fy'] = ins.fy
	return intrinsics

