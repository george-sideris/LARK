import cv2
import numpy as np
import multiprocessing as mp
import time
from pynput import keyboard
import os
import pyigtl
# import skvideo.io

cams = [1,2,3,4,5]
frameRate = 10
FROM_FILE = True  # use this to trim or edit speed of videos stored in srcFolder while keeping them synced
TRIM = True # trims video before firstFrame and after lastFrame
DISPLAY_ONLY = False  # disables recording to file
IGT = True
IGT_Host = "192.168.3.15"
IGT_Port = 18948
# IGT_offset = (195.41802150, -62.89397333, -21.33193215)  # T9
# IGT_offset = (159.27860385, -49.17121679, -23.86903035)  # landmark tests
# IGT_offset = (1.23028963, -0.37329041, -0.26733674)  # medtronic
# IGT_offset = (158.47052178, -49.13006386, -23.98885010)  # T26
# IGT_offset = (167.43914233, 1.74584740, -23.23567740)  # T29
# IGT_offset = (166.64971829, 1.66082794, -23.64581970)  # T33
# IGT_offset = (166.97884775, 1.40624628, -22.90124877)  # T35
IGT_offset = (166.93284598, 1.92530048, -23.27872542)
srcFolder = 'HT03_RF'
firstFrame = 0
lastFrame = 160

def runCam(cam, filePath, stopEvent, barrier):
    frameTime = 1/frameRate
    if FROM_FILE: cap = cv2.VideoCapture(f'{srcFolder}/{cam}.mp4')
    else: cap = cv2.VideoCapture(f"udpsrc address=192.168.3.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {cam}")
        return
    
    # Read the first frame to get the dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        cap.release()
        return
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    print(f"Cam {cam} size: {frame_height, frame_width}")
    
    if not DISPLAY_ONLY: 
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv2Writer = cv2.VideoWriter(f'{filePath}/{cam}.mp4', fourcc, frameRate, (frame_width, frame_height))
        # ffmpegWriter = skvideo.io.FFmpegWriter(f'{filePath}/{cam}.mp4', outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset':'veryslow'}) 
        
        if not cv2Writer.isOpened():
            print(f"Error: Cannot open video writer with file {filePath}")
            cap.release()
            return
    
    frameIndex = 1
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        # print(f'cam {cam} waiting')
        barrier.wait()
        # print(f'cam {cam} passed')
        
        # print(f'Camera {cam}, Frame {frameIndex}: {time.time()} s')
        ret, frame = cap.read()  # ret is True if frame is read correctly
        if not ret:
            print(f"Can't receive frame from camera {cam}.")
            break
        
        if not DISPLAY_ONLY and (not TRIM or (TRIM and firstFrame <= frameIndex <= lastFrame)): 
            cv2Writer.write(frame)
            # ffmpegWriter.writeFrame(frame[:,:,::-1])
        elif TRIM and frameIndex > lastFrame: stopEvent.set()
        cv2.imshow(f'Camera {cam}', frame)
        
        frameIndex += 1
        if cv2.pollKey() == ord('q'):
            stopEvent.set()

    cap.release()
    if not DISPLAY_ONLY: 
        cv2Writer.release()
        # ffmpegWriter.close()
    cv2.destroyWindow(f'Camera {cam}')
    print(f'cam {cam} process closed')
    
def IGTCapture(host, port, filePath, stopEvent, barrier, offset):
    client = pyigtl.OpenIGTLinkClient(host, port, start_now=True)
    offsetTransform = np.eye(4, dtype=np.float32)
    offsetTransform[:3, 3] = offset
    positions = []
    
    frameIndex = 1
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        # print('IGT waiting')
        barrier.wait()
        # print('IGT passed')
        message = client.wait_for_message(device_name="StylusToReference", timeout=1/frameRate-.01)
        if message is not None and (not TRIM or (TRIM and firstFrame <= frameIndex <= lastFrame)):
            transform = message.matrix
            position = (transform @ offsetTransform)[:3,3].reshape(1,3)
            positions.append(position)
    
    client.stop()
    positions_array = np.vstack(positions)
    np.savetxt(f'{filePath}/IGTPos.txt', positions_array)
    print('IGT process closed')
    
def onPress(key):
    global stopEvent
    try:
        if key.char == 'q':
            stopEvent.set()
    except AttributeError:
        pass
    
if __name__ == "__main__":
    filePath = f'{time.strftime("%Y-%m-%d_%H_%M_%S")}'
    frameTime = 1/frameRate
    lastFrameTime = -frameTime
    if not DISPLAY_ONLY: os.mkdir(filePath)
    processes = dict()
    stopEvent = mp.Event()
    if IGT: barrier = mp.Barrier(len(cams)+2, timeout=10)
    else: barrier = mp.Barrier(len(cams)+1, timeout=10)
    listener = keyboard.Listener(on_press=onPress)
    listener.start()
    
    for cam in cams:
        process = mp.Process(target=runCam, args=(cam, filePath, stopEvent, barrier))
        process.start()
        processes[cam] = process
        
    if IGT:
        IGTProcess = mp.Process(target=IGTCapture, args=(IGT_Host, IGT_Port, filePath, stopEvent, barrier, IGT_offset))
        IGTProcess.start()
    
    capIndex = 1
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if (currTime:=time.time()) - lastFrameTime >= frameTime:
            # print('main waiting')
            barrier.wait()
            # print('main passed')
            totalTime = capIndex*frameTime
            print(f"{int(totalTime/60)} min {int(totalTime%60)} s, {1/(currTime-lastFrameTime):.2f} fps, frame {capIndex}", end='\r')
            lastFrameTime = currTime
            capIndex += 1
            if TRIM and capIndex > lastFrame: stopEvent.set()
    print('main process ended')
        
