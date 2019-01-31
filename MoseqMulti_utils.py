import json, os, sys, time, pickle, h5py, cv2, numpy as np, subprocess, datetime

def load_metadata(PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_metadata.json'
    return json.load(open(fname))

def load_alignment(PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_alignment.h5'
    return h5py.File(fname,'r')


# simple command to pipe frames to an ffv1 file (adapted from moseq2-extract)
def write_depth_frames(filename, frames, threads=6, fps=30,
                 pixel_format='gray16le', codec='ffv1', close_pipe=True,
                 pipe=None, slices=24, slicecrc=1, frame_size=None, get_cmd=False):
    """
    Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])

    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(frames.shape[0]):
        pipe.stdin.write(frames[i,:,:].astype('uint16').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe



# simple command to pipe frames to an ffv1 file (adapted from moseq2-extract)
def write_color_frames(filename, frames, threads=6, fps=30, crf=13,
                 pixel_format='rgb24', codec='ffv1',close_pipe=True,
                 pipe=None,  slices=24, slicecrc=1, frame_size=None, get_cmd=False):
    """
    Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])


    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-threads', str(threads),
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               '-crf',str(crf),
               filename]


    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(frames.shape[0]):
        pipe.stdin.write(frames[i,:,:,:].astype('uint8').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe



# simple command to pipe frames to an ffv1 file (adapted from moseq2-extract)
def write_binary_frames(filename, frames, threads=6, fps=30, crf=5,
                 pixel_format='gray', codec='ffv1',close_pipe=True,
                 pipe=None,  slices=24, slicecrc=1, frame_size=None, get_cmd=False):
    """
    Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])


    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-threads', str(threads),
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               '-crf',str(crf),
               filename]


    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(frames.shape[0]):
        pipe.stdin.write((frames[i,:,:].astype('uint8')*255).tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe



def read_depth_frames(filename, frames, threads=6, fps=30,
                pixel_format='gray16le', frame_size=(640,480),
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .mp4/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices
    Returns:
        3d numpy array:  frames x h x w
    """

    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    video = np.frombuffer(out, dtype='uint16').reshape((len(frames), frame_size[1], frame_size[0]))
    return video


def read_color_frames(filename, frames, threads=6, fps=30,
                pixel_format='rgb24', frame_size=(640,480),
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .mp4/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices
    Returns:
        4d numpy array:  frames x h x w x 3
    """

    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    video = np.frombuffer(out, dtype='uint8').reshape((len(frames), frame_size[1], frame_size[0], 3))
    return video



def read_binary_frames(filename, frames, threads=6, fps=30,
                pixel_format='gray', frame_size=(640,480),
                slices=24, slicecrc=1, get_cmd=False):

    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    video = np.frombuffer(out, dtype='uint8').reshape((len(frames), frame_size[1], frame_size[0])) 
    return video


def load_color(frames, sn, PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_' + sn + '_color.mp4'
    return read_color_frames(fname, frames)

def load_depth(frames, sn, PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_' + sn + '_depth.avi'
    return read_depth_frames(fname, frames)


def load_color_mask(frames, sn, PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_' + sn + '_color_mask.avi'
    return read_binary_frames(fname, frames)

def load_depth_mask(frames, sn, PARAMS, session_name=None):
    if session_name is None: session_name = PARAMS['session_name']
    fname = PARAMS['working_directory'] + '/data/' + session_name + '_' + sn + '_depth_mask.avi'
    return read_binary_frames(fname, frames)

def get_aligned_frameset(frame_indexes, PARAMS):
    metadata = load_metadata(PARAMS)
    alignment = load_alignment(PARAMS)
    one_frame = False
    if type(frame_indexes)==int: 
        one_frame = True
        frame_indexes = [frame_indexes]
    color, depth = {},{}

    for ii,sn in enumerate(metadata['serial_numbers']):
        color[sn] = load_color(alignment['aligned_frames'][frame_indexes,ii], sn, PARAMS)
        depth[sn] = load_depth(alignment['aligned_frames'][frame_indexes,ii], sn, PARAMS)
        if one_frame:
            color[sn] = color[sn].squeeze()
            depth[sn] = depth[sn].squeeze()
    return color, depth





