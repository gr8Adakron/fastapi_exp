# import the necessary packages
import argparse
import base64
import json
import os
import random
import string
import time
from multiprocessing import Pool
from subprocess import PIPE, Popen

import cv2
import imutils
import numpy as np
import requests
from imutils.video import FPS, FileVideoStream

from cfunctions import *

def RandomString():
	rand_om = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(7))
	return str(rand_om)

def distant_model(api_input):
    request_address         = "http://127.0.0.1:8000/request"
    request_dict            = {"input":api_input}
    recieved_output 		= requests.post(request_address, json=request_dict, timeout=60)
    recieved_output_json 	= dict(json.loads(recieved_output.text))
    ann_batch_output        = recieved_output_json["output"]
    return ann_batch_output

def saving_batch_ann(output_dir,ann_batch_output):
    output_list = ann_batch_output
    for single_output in output_list:
        _id = single_output["id"]
        image_base64 = single_output["annotated_image"]
        save_path    = f"{output_dir}/{_id}.jpeg"
        save_base64_to_file(image_base64,save_path)

def form_individual_batches(messenger_dict):
    input_dir   = messenger_dict["input_dir"]
    output_dir  = messenger_dict["output_dir"]
    batch_input = messenger_dict["batch_input"]
    api_input   = []
    for single_image in batch_input:
        _id = single_image.split(".")[0]
        full_image_path = f"{input_dir}/{single_image}"
        base64_image  = read_file_as_base64(full_image_path).decode("utf-8")
        api_input.append({
            "id": _id,
            "image": base64_image
        })
    ann_batch_output  = distant_model(api_input)
    saving_batch_ann(output_dir,ann_batch_output)

def parallelize_calls(workers_msg_list,async_workers_count):
    with Pool(processes = async_workers_count) as pool:
        pool.map(form_individual_batches, workers_msg_list)
    pool.close()

def return_io_frames_holder():
	in_base_path   = "video_processing/inputs/"
	op_base_path   = "video_processing/outputs/"
	holder_name	= RandomString()
	in_holder_path = f"{in_base_path}{holder_name}"
	op_holder_path = f"{op_base_path}{holder_name}"
	os.mkdir(in_holder_path)
	os.mkdir(op_holder_path)
	return in_holder_path,op_holder_path

def colonel_batch(input_dir,output_dir,batch_size,async_workers_count):
    image_list	     = os.listdir(input_dir)
    total_image	     = len(image_list)
    no_of_iteration  = int(total_image/batch_size)+2
    start_time	     = time.time() 
    all_input	     = []
    all_img_names    = []
    all_output	     = {}
    asyn_worker_list = []
    start_time       = time.time()
    for current_batch in range(1,no_of_iteration):
        batch_begin  = int(batch_size*(current_batch-1))
        batch_end    = int(batch_size*(current_batch))

        if(batch_end>total_image): batch_end=total_image
        if(batch_begin==batch_end): break;
        batch_input  = image_list[batch_begin:batch_end]
        asyn_worker_list.append({
            "input_dir":input_dir,
            "output_dir": output_dir,
            "batch_input":batch_input
        })
    print(f"\t - Calling Human Detection model using Batch Size: {batch_size}, with Async workers: {async_workers_count}")
    parallelize_calls(asyn_worker_list,async_workers_count)
    timetaken  = format(float((time.time() - start_time)),'.3f')
    print(f"\t - TimeTaken for {total_image} frames {timetaken} sec")

def write_detected_video(output_video_path,op_holder_path):
	img_array = []
	for filename in sorted(glob.glob(f'{op_holder_path}/*.jpeg')):
	    image = cv2.imread(filename)
	    height, width, layers = image.shape
	    size = (width,height)
	    img_array.append(image)

	out = cv2.VideoWriter(f"{output_video_path}",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
	 
	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()

def main(input_video_path,output_video_path,batch_size,async_workers_count):

	frame_id   = 0
	video_path = 'sample_video/car_stopping_shorter_version.mp4'
	fvs		= FileVideoStream(input_video_path).start()
	time.sleep(1.0)
	fps		= FPS().start()
	in_holder_path,op_holder_path = return_io_frames_holder()
	print(in_holder_path,op_holder_path)
	# loop over frames from the video file stream
	input_frames_list = []
	while fvs.more():
		frame_id+=1
		frame = fvs.read()
		if(frame is None):
			continue

		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
		cv2.imwrite(f"{in_holder_path}/{frame_id}.jpeg",frame)
		

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()
	colonel_batch(in_holder_path,op_holder_path,batch_size,async_workers_count)
	write_detected_video(output_video_path,op_holder_path)

if __name__ == '__main__':
	input_video_path    = 'sample_video/car_stopping_shorter_version.mp4'
	output_video_path   = 'output.avi'
	batch_size          = 1
	async_workers_count = 2
	main(input_video_path,output_video_path,batch_size,async_workers_count)