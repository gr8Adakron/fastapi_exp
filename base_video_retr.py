import base64
import io
import json
import time
from io import BytesIO

import cv2
import numpy as np
import requests
# import the necessary packages
from imutils.video import FPS, FileVideoStream
from PIL import Image


def convert_numpy2base64(arr):
	im            = Image.fromarray(arr.astype("uint8"))
	rawBytes      = io.BytesIO()
	im.save(rawBytes, "PNG")
	rawBytes.seek(0)  # return to the start of the file
	base64_string = base64.b64encode(rawBytes.read())
	base64_string = base64_string.decode('utf-8')
	return base64_string

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def convert_base642numpy(b64_string):
	image  =  Image.open(BytesIO(base64.b64decode(b64_string)))
	image  =  toRGB(image)
	return image

def distant_model(base64_frame,frame_id,request_address):
	input_dict              = {}
	input_dict["image"]     = base64_frame
	input_dict["id"]        = frame_id
	request_dict            = {"input":[input_dict]}
	recieved_output 		= requests.post(request_address, json=request_dict)
	recieved_output_json 	= dict(json.loads(recieved_output.text))
	ann_base64_frame   = recieved_output_json["output"][0]["annotated_image"]
	return ann_base64_frame

def stream_frames_to_model(frame,frame_id):
	base64_frame          = convert_numpy2base64(frame)
	request_address       = "http://127.0.0.1:8000/request"
	ann_base64_frame      = distant_model(base64_frame,frame_id,request_address)
	ann_frame             = convert_base642numpy(ann_base64_frame)
	return ann_frame

# def main(video_path):
# 	cap     = cv2.VideoCapture(video_path)
# 	number  = 0
# 	while(cap.isOpened()):
# 		number+=1
# 		print(number)
# 		ret, original_frame   = cap.read()
# 		if(original_frame is None):
# 			continue
# 		ann_frame             = stream_frames_to_model(original_frame,number)
# 		cv2.imshow('frame',ann_frame)
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break
# 		print(number)

# 	cap.release()
# 	cv2.destroyAllWindows()

def main(video_path):
	video_path = 'sample_video/car_stopping_shorter_version.mp4'
	fvs        = FileVideoStream(video_path).start()
	time.sleep(1.0)
	fps        = FPS().start()

	# loop over frames from the video file stream
	input_frames_list = []
	number = 0
	while fvs.more():
		number+=1
		frame = fvs.read()
		if(frame is None):
			continue

		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
		
		ann_frame             = stream_frames_to_model(frame,number)
		cv2.imshow('frame',ann_frame)
		cv2.waitKey(1)
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()

if __name__ == '__main__':
	video_path = 'sample_video/car_stopping_shorter_version.mp4'
	main(video_path)
