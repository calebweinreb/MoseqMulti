import json, os, sys, time, pickle, h5py, cv2, numpy as np, multiprocessing as mp, matplotlib.pyplot as plt
import cv2.aruco as aruco
from functools import partial
from MoseqMulti_utils import *

def get_checkpoints(img,origin):
    N = 4
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (N,N),None)
    if ret:
        points1 = corners.squeeze()
        points2 = np.rot90(points1.reshape((N,N,2)),axes=(0,1)).reshape(16,2)
        points3 = np.rot90(points2.reshape((N,N,2)),axes=(0,1)).reshape(16,2)
        points4 = np.rot90(points3.reshape((N,N,2)),axes=(0,1)).reshape(16,2)  
        d1 = np.sqrt(((points1[0,:] - origin)**2).sum())
        d2 = np.sqrt(((points2[0,:] - origin)**2).sum())
        d3 = np.sqrt(((points3[0,:] - origin)**2).sum())
        d4 = np.sqrt(((points4[0,:] - origin)**2).sum())
        return [points1,points2,points3,points4][int(np.argmin([d1,d2,d3,d4]))]
    return []


def corner_crop(img,cc):
    # for each direction, find the lowest / highest shape (hopefully corners)
    # and crop to the top / bottom of that shape
    max_shape = np.argmax(cc[:,:,0].max(1))
    x_max = int(cc[max_shape,:,0].min())
    #x_max_y = int(cc[max_shape,np.argmin(cc[max_shape,:,0]),1]
    
    max_shape = np.argmax(cc[:,:,1].max(1))
    y_max = int(cc[max_shape,:,1].min())
    #y_max_x = int(cc[max_shape,np.argmin(cc[max_shape,:,1]),0])
    
    min_shape = np.argmin(cc[:,:,0].min(1))
    x_min = int(cc[min_shape,:,0].max())
    #x_min_y = int(cc[min_shape,np.argmax(cc[max_shape,:,0]),1]
    
    min_shape = np.argmin(cc[:,:,1].min(1))
    y_min = int(cc[min_shape,:,1].max())
    #y_min_x = int(cc[min_shape,np.argmax(cc[max_shape,:,1]),0])
    
    x_min = int(np.max([x_min-50,0]))  
    x_max = int(np.min([x_max+50,img.shape[1]]))  
    y_min = int(np.max([y_min-50,0]))  
    y_max = int(np.min([y_max+50,img.shape[0]]))  
    
    return img[y_min:y_max,x_min:x_max], x_min, y_min


def detect_keypoints(img,depth):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    key_points = {}
    if not (ids is None):
        ids = np.array(ids).squeeze()
        if len(ids.shape) > 0:
            ids = ids[np.in1d(ids,range(56))]   
            labels = np.array([id_labels[i] for i in ids])
            used_labels = np.nonzero(np.bincount(labels) >= 4)[0]
            if len(used_labels) > 0:
                try:
                    for l in used_labels:
                        cc = np.zeros((8,4,2))
                        cc[ids[labels==l]-l*8,:,:] = np.array(corners).squeeze()[labels==l,:,:]
                        checker_img, x_min, y_min = corner_crop(np.array(img),cc[ids[labels==l]-l*8,:,:])
                        origin = cc[0,0,:] - np.array([x_min,y_min])
                        if np.sum(cc[0,0,:] != 0) > 0:
                            points = get_checkpoints(checker_img, origin)
                            if len(points) > 0:
                                key_points[l] = points
                                key_points[l][:,0] += x_min
                                key_points[l][:,1] += y_min
                                key_points[l] = np.vstack((key_points[l],cc.reshape(32,2)))
                                d = [depth[int(key_points[l][ii,1]),int(key_points[l][ii,0])] for ii in range(key_points[l].shape[0])]
                                key_points[l] = np.hstack((key_points[l],np.array(d).reshape((len(d),1))))

                except: pass   
    return key_points


def deproject(pixel,depth,intrinsics):
    x = (pixel[:,0] - intrinsics['ppx']) / intrinsics['fx']
    y = (pixel[:,1] - intrinsics['ppy']) / intrinsics['fy']
    return np.vstack((depth * x, depth * y, depth)).T 
    #return np.vstack((pixel.T, depth)).T 
    
def transform_key_points(all_kp, intrinsics):
    out = []
    for kp in all_kp:
        new_kp = {}
        for l in kp:
            new_kp[l] = deproject(kp[l][:,:2],kp[l][:,2], intrinsics)
        out.append(new_kp)
    return out


def filter_key_points(all_kp):
    mask = np.zeros((6,len(all_kp),48))
    kps = np.zeros((6,len(all_kp),48,3))
    for label in range(6):
        ff = np.array([label in kp for kp in all_kp])
        nz = np.nonzero(ff)[0]
        if len(nz) > 0:
            X = np.array([all_kp[i][label] for i in nz])
            mask[label,ff,:] = np.all(np.abs(X) > .001, axis=2)
            mask[label,ff,:] = mask[label,ff,:] * (X[:,:,2] < (np.median(X[:,:,2]) + 500))
            kps[label,ff,:] = X
    return kps,mask


def rigid_transform_3D(A, B):
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA).dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    t = -R.dot(centroid_A.T) + centroid_B.T
    return R, t

def get_transforms(PARAMS):
    metadata = load_metadata(PARAMS)
    serial_numbers = metadata['serial_numbers']
    keypoints = get_all_keypoints(PARAMS)
    all_key_points_realspace = [transform_key_points(keypoints[i], metadata['intrinsics'][serial_numbers[i]]) for i in range(len(keypoints))]
    kp_mats, masks = zip(*[filter_key_points(all_key_points_realspace[i]) for i in range(len(keypoints))])
    
    MMs = [np.swapaxes(masks[i], 1,0).reshape(masks[i].shape[1],6*48) for i in range(len(keypoints))]
    fig,axs = plt.subplots(1,len(keypoints))
    for i in range(len(keypoints)):
        axs[i].imshow(np.repeat(MMs[i],5,axis=1))
        axs[i].set_title(serial_numbers[i])
    fig.set_size_inches((15,8))

    reference_camera = PARAMS['reference_camera']
    transforms = {reference_camera:(np.identity(3),np.zeros(3))}
    for ii,sn in enumerate(serial_numbers):
        if sn != reference_camera:
            transforms[sn] = get_transform(kp_mats, masks, ii,serial_numbers.index(reference_camera))
    return transforms
 


def get_keypoints_one_camera(cam_ix, PARAMS):
    keypoints = []
    serial_numbers = load_metadata(PARAMS)['serial_numbers']
    frames_ix = load_alignment(PARAMS)['aligned_frames'][:,cam_ix]
    path_prefix = PARAMS['working_directory']+'/data/'+PARAMS['session_name']+'_'+serial_numbers[cam_ix]
    for i in range(0,len(frames_ix),PARAMS['batch_size']):
        print('Processing frame',i,'out of',np.max(frames_ix),'for camera',cam_ix)
        frame_batch = frames_ix[i:i+PARAMS['batch_size']]
        color = read_color_frames(path_prefix+'_color.mp4', frame_batch)
        depth = read_depth_frames(path_prefix+'_depth.avi', frame_batch)
        for ii in range(len(frame_batch)):
            keypoints.append(detect_keypoints(color[ii,...],depth[ii,...]))
    return np.array(keypoints)   

def get_all_keypoints(PARAMS):
    serial_numbers = load_metadata(PARAMS)['serial_numbers']
    get_keypoints_one_camera_mappable = partial(get_keypoints_one_camera, PARAMS=PARAMS)
    if PARAMS['multiprocessing']:
        return mp.Pool().map(get_keypoints_one_camera_mappable, range(len(serial_numbers)))
    else:
        return [get_keypoints_one_camera_mappable(ii) for ii in range(len(serial_numbers))]

        
def get_transform(kp_mats, masks, i1,i2):
    points1 = []
    points2 = []
    for t in range(masks[i1].shape[1]):
        ff1 = masks[i1][:,t,:].flatten()
        ff2 = masks[i2][:,t,:].flatten()
        ff = np.all([ff1,ff2],axis=0)
        if ff.sum() > 0:
            points1.append(kp_mats[i1][:,t,:,:].reshape(len(ff),3)[ff,:])
            points2.append(kp_mats[i2][:,t,:,:].reshape(len(ff),3)[ff,:])
    if len(points1)==0:
        print('Alert! no common keypoints for cameras',i1,'and',i2)
        return None
    
    points1 = np.vstack(points1)
    points2 = np.vstack(points2)
    print('Transforming camera',i1,'to camera',i2,'with',points1.shape[0],'points')

    ff1 = np.sqrt(((points1-points1.mean(0))**2).sum(1)) < 120
    ff2 = np.sqrt(((points2-points2.mean(0))**2).sum(1)) < 120
    ff = np.all([ff1,ff2],axis=0)
    points1 = points1[ff,:]
    points2 = points2[ff,:]
    R,t = rigid_transform_3D(points1, points2)
    return R,t


id_labels = {}
for pattern in range(6):
    for i in range(8):
        id_labels[i+pattern*8] = pattern










