import shutil, json, os, sys, time, pickle, h5py, cv2, numpy as np, matplotlib.pyplot as plt, h5py
from MoseqMulti_utils import *
from skimage.measure import label
from skimage.filters import threshold_otsu
from functools import partial

def segment(PARAMS, color_background, depth_background):
    serial_numbers = load_metadata(PARAMS)['serial_numbers']
    segment_one_camera_mappable = partial(segment_one_camera, PARAMS=PARAMS, 
                                          color_background=color_background, 
                                          depth_background=depth_background)
    if PARAMS['multiprocessing']:
        mp.Pool().map(segment_one_camera_mappable, serial_numbers)
    else: 
        for sn in serial_numbers: segment_one_camera_mappable(sn)
        
        
def segment_one_camera(sn, PARAMS, color_background, depth_background):
    print('Segmenting camera',sn)
    num_frames = load_metadata(PARAMS)['num_frames']
    color_bg = color_background[sn]
    depth_bg = depth_background[sn]
    color_mask_pipe = None
    depth_mask_pipe = None
    path_prefix = PARAMS['working_directory']+'/data/'+PARAMS['session_name']+'_'+sn
    
    for i in range(0,num_frames, PARAMS['batch_size']):
        print('...on frame',i,'out of',num_frames)
        frames = range(i,np.min([i+PARAMS['batch_size'], num_frames]))
        color_data = load_color(frames, sn, PARAMS)
        depth_data = load_depth(frames, sn, PARAMS)
        color_mask, depth_mask = [],[]
        for current_frame in range(color_data.shape[0]):
            Xcolor = color_data[current_frame,:,:,:]
            Xdepth = depth_data[current_frame,:,:]
            color_mask.append(clean_color_seg(color_seg(Xcolor,color_bg), Xcolor, PARAMS))
            depth_mask.append(np.all([color_mask[-1],depth_seg(Xdepth,depth_bg, PARAMS)],axis=0))
            if i > PARAMS['skip_first_frames']: color_bg, depth_bg = update_background(color_bg, depth_bg, Xcolor, Xdepth, color_mask[-1], PARAMS['background_update_rate'])
        
        color_mask_pipe = write_binary_frames(path_prefix+'_color_mask.avi', np.array(color_mask), 
                                              pipe=color_mask_pipe, close_pipe=False)
        
        depth_mask_pipe = write_binary_frames(path_prefix+'_depth_mask.avi', np.array(depth_mask), 
                                              pipe=depth_mask_pipe, close_pipe=False)
        
    color_mask_pipe.stdin.close()
    depth_mask_pipe.stdin.close()


def color_seg(x,bg):
    try:
        y = np.array(np.abs((x/x.sum(2)[:,:,None] - bg/bg.sum(2)[:,:,None])).sum(2),dtype=float) * bg.sum(2)
        th =  threshold_otsu(y)
        th = np.max([th, 22])
        t1 = y > th

    except: t1 = np.zeros(x.shape[:2])
    y = np.array(np.abs((x - bg)).sum(2),dtype=float)
    th = threshold_otsu(y)
    t2 = y > th
    return np.any([t1,t2],axis=0)


def depth_seg(x,bg, PARAMS):
    y = (bg-x) * (x != 0)
    y = cv2.GaussianBlur(y,(11,11),0)
    y = (y > PARAMS['min_depth_difference']) * (y < PARAMS['max_depth_difference']) * (x != 0)
    return y


def color_seg(x,bg):
    try:
        y = np.array(np.abs((x/x.sum(2)[:,:,None] - bg/bg.sum(2)[:,:,None])).sum(2),dtype=float) * bg.sum(2)
        th = threshold_otsu(y)
        t1 = y > th
    except: t1 = np.zeros(x.shape[:2])
    y = np.array(np.abs((x - bg)).sum(2),dtype=float)
    th = threshold_otsu(y)
    t2 = y > th
    return np.any([t1,t2],axis=0)

        
def clean_color_seg(mask, Xcolor, PARAMS):
    ccs = label(mask)
    out = np.zeros(mask.shape)

    for k in np.nonzero(np.bincount(ccs.flatten()) > PARAMS['min_color_component_size'])[0][1:]:
        nz1 = np.nonzero((ccs==k).sum(0)>0)[0]
        nz2 = np.nonzero((ccs==k).sum(1)>0)[0]
        aspect_ratio = np.max([(nz1.max()-nz1.min()), (nz2.max()-nz2.min())])**2 / (ccs==k).sum()
        if aspect_ratio < PARAMS['max_color_component_aspect_ratio']:
            brightness = (Xcolor*np.array(ccs==k,dtype=float)[:,:,None]).mean() / (ccs==k).mean()
            if brightness < PARAMS['max_color_component_brightness']:
                out += (ccs==k)
            
    ccs = label(out==0)
    for k in np.nonzero(np.bincount(ccs.flatten()) < PARAMS['max_color_component_island_size'])[0]:
        out += (ccs==k)
    return np.array(out > 0,dtype=float)

def update_background(color_bg, depth_bg, Xcolor, Xdepth, color_mask, rate):
    bg_mask = (cv2.GaussianBlur(color_mask,(71,71),0) > .02)
    color_bg = color_bg*bg_mask[:,:,None]*.1 + Xcolor*(bg_mask==0)[:,:,None]*.1 + color_bg*.9  

    bg_mask = np.any([bg_mask, Xdepth == 0], axis=0) 
    depth_bg = depth_bg*bg_mask*rate + Xdepth*(bg_mask==0)*rate + depth_bg*(1-rate)  
    return color_bg, depth_bg

def get_depth_background(depth_data):
    depth_data = np.array(depth_data,dtype=float)
    for i in range(0,depth_data.shape[0],20):
        for j in range(depth_data.shape[1]):
            for k in range(depth_data.shape[2]):
                if depth_data[i,j,k] == 0: depth_data[i,j,k] = np.nan
    return np.nan_to_num(np.nanmedian(depth_data,axis=0))
        
def get_color_background(color_data):
    return np.median(color_data,axis=0)

def get_background(PARAMS):
    serial_numbers = load_metadata(PARAMS, session_name=PARAMS['background_session_name'])['serial_numbers']
    frame_ix = range(PARAMS['skip_first_frames'],PARAMS['skip_first_frames']+PARAMS['n_background_frames'])
    color_background, depth_background = {},{}
    for sn in serial_numbers:
        print('Getting background for camera',sn)
        color_background[sn] = get_color_background(load_color(frame_ix, sn, PARAMS, session_name=PARAMS['background_session_name']))
        depth_background[sn] = get_depth_background(load_depth(frame_ix, sn, PARAMS, session_name=PARAMS['background_session_name']))
    return color_background, depth_background
