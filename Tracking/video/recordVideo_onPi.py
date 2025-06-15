import time
from pynput import keyboard
import subprocess
import os
import paramiko
import multiprocessing as mp

cams = [1,2,3,4,5]  # First cam must be the SOURCE cam and all others must be SINK cams
frameRate = 10
frames = 600  # Record a specific number of frames. Set to 0 to record indefinitely.

user = 'nist'
sshKeys = dict([(cam, os.path.expanduser(f'~/.ssh/rpi_cam{cam}')) for cam in cams])

def runCam(cam, barrier, filePath, source):
    if source:
        if frames: videoCmd = f'rpicam-vid -t 0 -n -k --initial pause --inline --framerate {frameRate} --frames {frames+1} --width 1456 --height 1088 --intra 1 --level 4.2 --profile high --bitrate 100000000 --denoise cdn_off -o {filePath}.h264'
        else: videoCmd = f'rpicam-vid -t 0 -n -k --initial pause --inline --framerate {frameRate} --width 1456 --height 1088 --intra 1 --level 4.2 --profile high --bitrate 100000000 --denoise cdn_off -o {filePath}.h264'
    else:
        if frames: videoCmd = f'rpicam-vid -t 0 -n -k --initial record --inline --framerate {frameRate} --frames {frames} --width 1456 --height 1088 --intra 1 --level 4.2 --profile high --bitrate 100000000 --denoise cdn_off -o {filePath}.h264'
        else: videoCmd = f'rpicam-vid -t 0 -n -k --initial record --inline --framerate {frameRate} --width 1456 --height 1088 --intra 1 --level 4.2 --profile high --bitrate 100000000 --denoise cdn_off -o {filePath}.h264'
    
    sshClient = paramiko.SSHClient()
    sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sshClient.connect(hostname=f'cam{cam}.local', username=user, key_filename=sshKeys[cam])
    channel = sshClient.get_transport().open_session()
    channel.exec_command(videoCmd)
    
    if source:
        barrier.wait()
        channel.send('\n')  # Start recording
    
    if not frames:
        barrier.wait()
        channel.send('x\n')  # Stop recording
    
    while True:
        if channel.exit_status_ready(): break
    time.sleep(5)
    
    subprocess.call(f'scp -i {sshKeys[cam]} {user}@cam{cam}.local:~/{filePath}.h264 {filePath}/{cam}.h264', shell=True)
    if not source: 
        subprocess.call(f'ffmpeg -framerate {frameRate} -i {filePath}/{cam}.h264 -c copy {filePath}/{cam}.mp4', shell=True)
    else: 
        subprocess.call(f'ffmpeg -framerate {frameRate} -i {filePath}/{cam}.h264 -c copy {filePath}/{cam}-untrimmed.mp4', shell=True)
        subprocess.call(f'ffmpeg -i {filePath}/{cam}-untrimmed.mp4 -vf select="gte(n\, 1)" {filePath}/{cam}.mp4', shell=True)
        # subprocess.call(f'rm {filePath}/{cam}-untrimmed.mp4', shell=True)
    # subprocess.call(f'rm {filePath}/{cam}.h264', shell=True)
    sshClient.exec_command(f'rm {filePath}.h264')
    sshClient.close()
    barrier.wait()

def onPress(key):
    global stop
    try:
        if key.char == 'q':
            stop = True
            print("\nStopping capture.")
    except AttributeError:
        pass

if __name__ == "__main__":
    stop = False
    filePath = f'{time.strftime(f"%Y-%m-%d_%H_%M_%S")}'
    os.mkdir(filePath)
    sourceBarrier = mp.Barrier(2)
    sinkBarrier = mp.Barrier(len(cams))
    sinkProcesses = dict()
    listener = keyboard.Listener(on_press=onPress)
    listener.start()

    for cam in cams[1:]:
        process = mp.Process(target=runCam, args=(cam, sinkBarrier, filePath, False))
        process.start()
        sinkProcesses[cam] = process
    
    sourceProcess = mp.Process(target=runCam, args=(cams[0], sourceBarrier, filePath, True))
    sourceProcess.start()
    time.sleep(3)
    sourceBarrier.wait()
    startTime = time.time()
    while not stop:
        time.sleep(.1)
        totalTime = time.time() - startTime
        print(f"{int(totalTime/60)} min {int(totalTime%60)} s", end='\r')
        if frames and totalTime >= frames / frameRate:
            stop = True
            time.sleep(3)

    if not frames:
        sourceBarrier.wait()  # Stop recording
        time.sleep(.5)
        sinkBarrier.wait()
        
    sourceBarrier.wait()  # Wait until copying is finished
    sinkBarrier.wait()
        