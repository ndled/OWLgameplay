import cv2
import math
import pandas as pd
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
import tensorflow as tf
from moviepy.editor import *
from keras.preprocessing.image import ImageDataGenerator
import subprocess
import matplotlib.pyplot as plt
import youtube_dl
import argparse

def idb_excepthook(type, value, tb):
    """Call an interactive debugger in post-mortem mode
    If you do "sys.excepthook = idb_excepthook", then an interactive debugger
    will be spawned at an unhandled exception
    """
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import pdb, traceback
        traceback.print_exception(type, value, tb)
        print
        pdb.pm()

def download(url): # add another arg to put this into special temo folder
    """This both downloads the mp4 AND returns the filename as a 
    string to be used later on.
    """
    ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio/best'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download= True)
        filename = ydl.prepare_filename(info)
    return filename

def videoCut(videoFile, imgname):
    count = 0
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            name ="" + imgname + str(count).zfill(5) + ".jpg";count+=1
            cv2.imwrite(name, frame)
        print("working on " + str(count), end='\r')
    cap.release()
    print("")
    print ("Done!")

def build_cnn(path_to_weights):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    cnn.compile(optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = [tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.FalsePositives()])
    cnn.load_weights(path_to_weights)
    return cnn

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def pred(cnn, dir):
    image_generator2 = ImageDataGenerator(rescale = 1./255)
    test_generator = image_generator2.flow_from_directory(batch_size=1,
                                                 directory=dir,
                                                 target_size=(64, 64), 
                                                 class_mode=None,
                                                 shuffle = False)

    pred_probs=cnn.predict_generator(test_generator, steps=len(test_generator), verbose=1)
    pred_bin=(pred_probs>0.5).astype("int32")
    test_pred = (moving_average(pred_bin.ravel(),60)>.5).astype("int32")

    # create list of time for gameplay

    times = []
    for i in range(0,len(test_pred)):
        if test_pred[i] == 1:
            times.append(i)

    # Turn that list into start stop pair

    count = 0
    start_stop = []
    for i in range(0,len(times)-1):
        if times[i] == times[i+1]-1:
            count = count + 1
        else:
            start_stop.append((times[i]-count,times[i]))
            count = 0 

    # Collapse any set of pairs that would represent a break in game play of less than 2 minutes

    reduced = [start_stop[0]]
    for t in start_stop:
        if t[0] - reduced[-1][1] < 120:
            reduced[-1] = (reduced[-1][0],t[1])
        else:
            reduced.append(t)

    #  Remove any gameplay clips less than 2 minutes.

    final_time = []
    for item in reduced:
        if item[1]-item[0] > 120:
            final_time.append(item)

    return final_time

def ffmpeg_time_format(final_time):
    select_text = "'"
    for i in range(0,len(final_time)):
        select_text = select_text+"between(t,"+str(final_time[i][0]-30)+","+str(final_time[i][1]+30)+")"
        if i < len(final_time)-1:
            select_text = select_text + "+"
    select_text = select_text + "'"
    return select_text

def ffmpeg_command(video,final_time,out):
    command = "ffmpeg -i \"{video}\" -vf \"select={select_text}, setpts=N/FRAME_RATE/TB\" -af \"aselect={select_text}, asetpts=N/SR/TB\" {out}.mp4".format(video=video, select_text=ffmpeg_time_format(final_time), out=out)
    return command

def  cleanup():
    pass

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "url", help="url you are processing")
    parser.add_argument(
        "--debug", "-d", action='store_true', help="Include debugging output")
    
    parsed = parser.parse_args()
    if parsed.debug:
        sys.excepthook = idb_excepthook

    #url = "https://youtu.be/uT3SBzmDxGk" 
    filename = download(parsed.url)
    temp_name = 'pred/images/7train_' # give this a more descriptive name, this is just an example
    #videoCut(filename, temp_name)

    path_to_weights = "model.h5" 
    cnn = build_cnn(path_to_weights)
    dir = 'pred/'

    command = ffmpeg_command(filename, pred(cnn,dir),'out')
    print(command)
    subprocess.call(command,shell=True)

if __name__ == "__main__":
    sys.exit(main(*sys.argv))

# def resolvepath(path): return os.path.realpath(os.path.normpath(os.path.expanduser(path)))