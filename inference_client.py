import os
import pathlib
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from io import StringIO

import cv2
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class LoadModel():
	def __init__(self):
		self.detection_model = self.load_model()
	
	def load_model(self):
		model_dir = "pretrained/human_detection/saved_model/1"
		model = tf.saved_model.load(str(model_dir))
		model = model.signatures['serving_default']
		return model

class InferenceHumanIdentifier(LoadModel):
	"""docstring for InferenceHumanIdentifier"""
	def __init__(self):
		super(InferenceHumanIdentifier, self).__init__()

	def run_inference_for_single_image(self,model, image_path_list):
		input2model = []
		output_dict = {}
		
		for image_path in image_path_list:
			image_path = str(image_path)
			image = load_img(image_path, target_size=(1024, 800))
			image = img_to_array(image,dtype=np.uint8)
			input2model.append(image)
			
		image = np.asarray(input2model)
		image.astype(np.uint8)
		input_tensor = tf.convert_to_tensor(image)
		
		output_dict_tensor_format = model(input_tensor)
		
		number_of_detections = output_dict_tensor_format['num_detections']
		for index,key in enumerate(list(output_dict_tensor_format.keys())):
			value = output_dict_tensor_format[key] 
			output_dict[key] = value.numpy()
		output_dict['num_detections'] = number_of_detections
		output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
		return output_dict

	def create_bb(self,detected_box,derived_scores,detection_classes,image_abs_path):
		image_abs_path = str(image_abs_path)
		output_save_path = image_abs_path
		img	  = cv2.imread(image_abs_path)
		h, w, _   = img.shape
		
		for index,box in enumerate(detected_box):
			class_detected = detection_classes[index]
			score	  = derived_scores[index]
			if(class_detected!=1):
				continue
			if(score<0.9):
				continue
			x1		 = int(box[1] * w)
			x2		 = int(box[3] * w)
			y1		 = int(box[0] * h)
			y2		 = int(box[2] * h)
			img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
			score	  = format(score,'.3f')
			bounding_box_image = cv2.putText(img, str(score), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			dest_folder = "outputs/"
			image_name  = image_abs_path.split("/")[-1]
			output_save_path = f"{dest_folder}{image_name}"
			cv2.imwrite(output_save_path,img)
		return output_save_path
			
	def postprocessing(self,detection_boxes,detection_scores,detection_classes,num_detections,image_path_list,image_id_list):
		return_list = []
		for index,image_abs_path in enumerate(image_path_list):
			_id					= image_id_list[index]
			valid_detection		= int(num_detections[index])
			indi_detection_classes = detection_classes[index][:valid_detection]
			indi_detection_boxes   = detection_boxes[index][:valid_detection]
			indi_detection_score   = detection_scores[index][:valid_detection]
			annotated_path		 = self.create_bb(indi_detection_boxes,indi_detection_score,indi_detection_classes,image_abs_path)
			return_list.append({
				"id":_id,
				"image_path": image_abs_path,
				"annotated_path": annotated_path,
				
			})
		return return_list

	def show_inference(self, model, image_path_list,image_id_list):
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		# Actual detection.
		output_dict		=  self.run_inference_for_single_image(model, image_path_list)
		detection_boxes	=  output_dict["detection_boxes"]
		detection_classes  =  output_dict["detection_classes"]
		detection_scores   =  output_dict["detection_scores"]
		num_detections	 =  output_dict["num_detections"]
		return_list		= self.postprocessing(detection_boxes,
							detection_scores,
							detection_classes,
							num_detections,
							image_path_list,
							image_id_list)
		return return_list

	def main(self,batch_of_image):
		batch_start = time.time()
		total_batch	 = len(batch_of_image)
		image_path_list = [single_img["image_path"] for single_img in batch_of_image]
		image_id_list   = [single_img["id"] for single_img in batch_of_image]
		return_list	 = self.show_inference(self.detection_model,image_path_list,image_id_list)
		batch_end	   = time.time()
		completion_time =  batch_end-batch_start
		print(f" > Batch of {total_batch} images took: {completion_time} .......... [OK]")
		return return_list
