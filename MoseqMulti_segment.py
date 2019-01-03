from skimage.measure import label
from skimage.filters import threshold_otsu
import shutil, json, os, sys, time, pickle, h5py, cv2, numpy as np, matplotlib.pyplot as plt, h5py

def get_depth_background(data, n_frames):
    serial_numbers = [sn.decode('UTF-8') for sn in data['frame_alignment']['serial_numbers']]
    depth_backgrounds = {}
    for sn in serial_numbers:
        print('Calculating background depth for camera',sn)
        Xdepth = np.array(data[sn]['depth'][::int(data[sn]['depth'].shape[0]/n_frames),:,:],dtype=float)
        for i in range(0,Xdepth.shape[0],20):
            for j in range(Xdepth.shape[1]):
                for k in range(Xdepth.shape[2]):
                    if Xdepth[i,j,k] == 0: Xdepth[i,j,k] = np.nan
        depth_backgrounds[sn] = np.nan_to_num(np.nanmedian(Xdepth,axis=0))
    return depth_backgrounds
        
def get_color_background(data, n_frames):
    serial_numbers = [sn.decode('UTF-8') for sn in data['frame_alignment']['serial_numbers']]
    color_backgrounds = {}
    for sn in serial_numbers:
        print('Calculating background color for camera',sn)
        Xcolor = np.array(data[sn]['color'][::int(data[sn]['depth'].shape[0]/n_frames),:,:,:],dtype=float)
        color_backgrounds[sn] = np.median(Xcolor,axis=0)
    return color_backgrounds

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
    t = time.time()
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
    return out

def update_background(color_bg, depth_bg, Xcolor, Xdepth, color_mask, rate):
    bg_mask = (cv2.GaussianBlur(color_mask,(71,71),0) > .02)
    color_bg = color_bg*bg_mask[:,:,None]*.1 + Xcolor*(bg_mask==0)[:,:,None]*.1 + color_bg*.9  

    bg_mask = np.any([bg_mask, Xdepth == 0], axis=0) 
    depth_bg = depth_bg*bg_mask*rate + Xdepth*(bg_mask==0)*rate + depth_bg*(1-rate)  
    return color_bg, depth_bg
    
'''
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
import numpy as np, matplotlib.pyplot as plt, scipy.sparse as ss
import cv2, os, h5py, sys, h5py, json, pickle
from skimage.measure import label 

def get_aligned_frames(data, metadata, serial_numbers, window):
    Ts = []
    for d,m in zip(data,metadata):
        T = d['timestamp'][:] - m['timestamps'][10]
        Ts.append(T)
        
    aligned_frames = []
    for i,t in enumerate(Ts[0]):
        frame = [i]
        for T in Ts[1:]:
            frame.append(np.argmin(np.abs(T - t)))
        times = [Ts[ii][frame[ii]] for ii in range(len(serial_numbers))]
        if (np.max(times)-np.min(times)) < window:
            aligned_frames.append(frame)
    return np.array(aligned_frames)

def load_background(data_dir, serial_numbers, session_id):
    bgdatas = {camera:np.load(data_dir+'/background_'+session_id+'_'+serial_numbers[camera]+'.npz') for camera in range(len(serial_numbers))}
    bg_depths = {}; bg_colors = {}
    for camera in bgdatas: 
        x = bgdatas[camera]['depth_median']
        d = np.percentile(x.flatten()[x.flatten() != 0],80)
        x = x + (x==0)*d
        bg_depths[camera] = x
        bg_colors[camera] = np.array(bgdatas[camera]['color_median'] ,dtype=int)
    return bg_colors,bg_depths

    
def get_image_and_segmentation(camera, current_frame,data,bg_colors,bg_depths,aligned_frames):
    xcolor = data[camera]['color'][aligned_frames[current_frame,camera],:,:,:]
    cmask = color_seg(xcolor,bg_colors[camera])
    xdepth = data[camera]['depth'][aligned_frames[current_frame,camera],:,:]
    dmask = depth_seg(xdepth,bg_depths[camera])
    mask = cmask * dmask 
    mask = get_large_components(mask, 500)
    #mask = clip_depth_outliers(mask,xdepth)
    xmask = get_masked_vals(xcolor,mask) 
    return xcolor, xdepth, cmask, dmask, mask, xmask

def get_segmented_scene(current_frame,data,metadata,bg_colors,bg_depths,aligned_frames,projection_map,transforms):
    colors = []
    positions = []
    cam_labels = []
    for camera in [0,1,2,3,4]:
        xcolor, xdepth, cmask, dmask, mask, xmask = get_image_and_segmentation(camera, current_frame,data,bg_colors,bg_depths,aligned_frames)
        mm = cv2.GaussianBlur(mask,(7,7),10)
        mask = mm > .7
        X,Y = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
        pixel = np.vstack((X.flatten(),Y.flatten()))[:,mask.flatten()>0].T
        depth = xdepth.flatten()[mask.flatten()>0]
        pp = deproject(pixel,depth, get_intrinsics(metadata[camera]))
        if camera in projection_map:
            R,t = transforms[projection_map[camera]]
            pp = pp.dot(R.T) + t
        positions.append(pp)
        cc = xcolor.reshape(xcolor.shape[0]*xcolor.shape[1],3)[mask.flatten()>0,:]
        cc = np.minimum(cc*.8 + np.array(plt.cm.jet(camera/len(data)))[:3]*100, 255)
        colors.append(cc)
        cam_labels.append([camera]*cc.shape[0])

    colors = np.vstack(colors)
    positions = np.vstack(positions)
    cam_labels = np.hstack(cam_labels)
    return positions, colors, cam_labels

def crop_to_mouse(positions):
    centroid = positions.mean(0)
    rr = np.sqrt(((positions-centroid)**2).sum(1))
    max_rad = np.percentile(rr,75) * 3
    ff = rr < max_rad

    origin, eXYZ, positions_XYZ = get_body_axis_transform(positions[ff,:])
    rr = np.sqrt((positions_XYZ[:,1:]**2).sum(1))
    max_rad = np.percentile(rr,95) * 1.15
    ff2 = rr < max_rad

    ff_out = np.array(ff)
    ff_out[ff] = ff2
    return ff_out

def get_body_axis_transform(positions):
    
    origin = positions.mean(0)
    X = positions-origin
    pca = PCA(n_components=3)
    
    eX = pca.fit(X).components_[0,:]
    eX = eX / np.sqrt((eX**2).sum())
    
    # get eY by taking cross product with downward pointing vector
    down = np.array([0,0,-1])
    eY = np.cross(eX,down)
    eY = eY / np.sqrt((eY**2).sum())
    
    eZ = np.cross(eY,eX)
    eZ = eZ / np.sqrt((eZ**2).sum())
    
    eXYZ = np.vstack((eX,eY,eZ)).T
    positions_XYZ = X.dot(eXYZ)
    return origin, eXYZ, positions_XYZ




def deproject(pixel,depth,intrinsics):
    x = (pixel[:,0] - intrinsics['ppx']) / intrinsics['fx']
    y = (pixel[:,1] - intrinsics['ppy']) / intrinsics['fy']
    return np.vstack((depth * x, depth * y, depth)).T 


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

    # special reflection case
    if np.linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T.dot(U.T)

    t = -R.dot(centroid_A.T) + centroid_B.T
    return R, t


def color_seg(x,bg):
    #return bg.sum(2)
    y = cv2.GaussianBlur(x,(11,11),0)
    y = np.array(np.abs((x - bg)).sum(2),dtype=float)
    #y = np.minimum(y,5)
    th = threshold_otsu(y)
    return y > th

def depth_seg(x,bg):
    y = (bg-x) * (x != 0)
    y = cv2.GaussianBlur(y,(11,11),0)
    y = (y > 10) * (y < 500) * (x != 0)
    return y

def get_largest_component(x):
    if np.sum(x != 0) == 0: return x
    ccs = label(x)
    k = np.argmax(np.bincount(ccs.flatten())[1:]) + 1
    return ccs == k

def get_large_components(x,thresh):
    if np.sum(x != 0) == 0: return x
    ccs = label(x)
    out = np.zeros(x.shape)
    for k in set(ccs.flatten()):
        if k != 0:
            if (ccs==k).sum() > thresh:
                out += (ccs==k)
    return out
            

def get_masked_vals(x,mask):
    mask3 = np.repeat(mask[:,:,None],3,axis=2)
    return np.array(np.minimum(x*1.5,255) * mask3 + (1-mask3)*0,dtype=int)

def clip_depth_outliers(mask,xdepth):
    closest = np.percentile(xdepth.flatten()[mask.flatten()>0],5)
    farthest = np.percentile(xdepth.flatten()[mask.flatten()>0],95)
    clipping_depth = farthest + (farthest-closest) * .5
    return np.all([mask, xdepth < clipping_depth], axis=0)

def get_bins(positions, N_BINS):
    p0 = np.percentile(positions[:,0],0.5)
    dd = (np.percentile(positions[:,0],99.5) - p0) / N_BINS
    bin_ffs = [np.all([positions[:,0] >= dd*i+p0, positions[:,0] <= dd*(i+1)+p0],axis=0) for i in range(N_BINS)]
    bins = [positions[ff,:] for ff in bin_ffs]
    return bins, bin_ffs

def get_spine(bins, spine_BEND):
    N_BINS = len (bins)
    #centroids = np.array([b.mean(0) for b in bins])
    centroids = []
    for b in bins:
        xmin = np.percentile(b[:,1],2)
        xmax = np.percentile(b[:,1],98)
        ymin = np.percentile(b[:,2],2)
        ymax = np.percentile(b[:,2],98)
        centroids.append((b[:,0].mean(),(xmin+xmax)/2, (ymin+ymax)/2))
    centroids = np.array(centroids)

    A = np.identity(N_BINS)
    A[1:,:-1] += np.identity(N_BINS-1) * (-1/2)
    A[:-1,1:] += np.identity(N_BINS-1) * (-1/2)
    A[0,:] = 0; A[-1,:] = 0
    A = A / spine_BEND + np.identity(N_BINS) 
    y = np.linalg.solve(A, centroids[:,1:])
    return np.hstack((centroids[:,0][:,None],y))


def get_slice_radii(X, N_SLICES):
    slice_has_points = []
    slice_radii = []
    aa = np.arctan(X[:,1] / X[:,0]) + np.pi / 2
    aa[X[:,0] < 0] += np.pi
    for s in range(N_SLICES):
        slice_ff = np.all([aa>s*2*np.pi/N_SLICES, aa<(s+1)*2*np.pi/N_SLICES],axis=0)
        if slice_ff.sum() <= 20:
            slice_has_points.append(0)
            slice_radii.append(0)
        else:
            slice_has_points.append(slice_ff.sum())
            slice_radii.append(np.percentile(np.sqrt((X[slice_ff,:]**2).sum(1)),60))
    return np.array(slice_radii), np.array(slice_has_points)


def get_ribbing(bins,spine,N_SLICES,rib_BEND_long,rib_BEND_across):
    N_BINS = len(bins)
    A = np.zeros((N_BINS*N_SLICES,N_BINS*N_SLICES))
    all_slice_radii = []
    all_has_slice_points = []
    for i,b in enumerate(bins):
        X = b[:,1:] - spine[i,1:]
        slice_radii, has_slice_points = get_slice_radii(X, N_SLICES)
        all_slice_radii.append(slice_radii)
        all_has_slice_points.append(has_slice_points)
        for j in range(N_SLICES):
            A[i*N_SLICES+j,i*N_SLICES+((j+1)%N_SLICES)] = 1./rib_BEND_across / 2
            if i > 0:
                A[i*N_SLICES+j,(i-1)*N_SLICES+j] = 1./ rib_BEND_long / 2
    all_slice_radii = np.hstack(all_slice_radii)
    all_has_slice_points = np.hstack(all_has_slice_points)
    A = np.diag(A.sum(1)) - A
    A = A + np.diag(all_has_slice_points) 
    radii = np.linalg.solve(A, all_has_slice_points * all_slice_radii)
    ribbing = np.zeros((N_BINS,N_SLICES,3))
    for i in range(N_BINS):
        for j in range(N_SLICES):
            vec = np.array([np.cos((j+.5)/N_SLICES*2*np.pi-np.pi/2),np.sin((j+.5)/N_SLICES*2*np.pi-np.pi/2)])
            vec = vec / np.sqrt((vec**2).sum())
            ribbing[i,j,1:] = vec * radii[i*N_SLICES+j] + spine[i,1:]
            ribbing[i,j,0] = spine[i,0]
    return ribbing


get_intrinsics = lambda meta: {k:meta[k] for k in ['ppx','ppy','fx','fy']}
'''

