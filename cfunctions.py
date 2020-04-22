
import time
import multiprocessing as mp
import io
import re
import csv
import json
import datetime
import time
import os
import tempfile
import base64
import requests, operator,time,unicodedata
import gc
import threading
import random,string
import cv2

def RandomString():
	rand_om = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(25))
	return str(rand_om)

def save_base64_to_file(image_data,save_path):
	image_data   = base64.b64decode(image_data)
	with open(f"{save_path}", "wb") as fh:
		fh.write(image_data)

def read_file_as_base64(image_path):
	with open(f"{image_path}", "rb") as image_file:
		image_data = base64.b64encode(image_file.read())
	return image_data


def validate_base64(input_batch):
	input_image_folder  = "input_images/"
	model_input        = []
	error_input        = []
	for single_image in input_batch:
		unq_file_name      = RandomString()
		single_image       = dict(single_image)
		image_data         = single_image.get('image')
		_id                = single_image.get('id')
		save_path          = f"{input_image_folder}{unq_file_name}.jpeg"
		
		try:
			save_base64_to_file(image_data,save_path)
			if(cv2.imread(save_path) is not None):
				pass
			else:
				error_input.append(_id)
				continue
			if((os.path.getsize(save_path)/1000000)>5):
				error_input.append(_id)
				continue
			else:
				pass
		except:
			error_input.append(_id)
			continue
		model_input.append({'image_path':save_path,'id':_id})
	return model_input,error_input

def post_processing(model_output):
	return_format      = []
	for single_result in model_output:
		annotated_path = single_result.get('annotated_path')
		_id            = single_result.get('id')
		image_data     = read_file_as_base64(annotated_path)
		return_format.append({
				"id": _id,
				"annotated_image":image_data,
			})
	return return_format