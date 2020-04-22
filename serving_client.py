import itertools
import operator
import os
import random
import string
import time
from glob import iglob
from pprint import pprint

import cv2
import numpy as np
import scipy
import tensorflow as tf
from grpc.beta import implementations
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from cfunctions import *
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


class ServingHumanIdentifier():

	def __init__(self):
		super(ServingHumanIdentifier, self).__init__()
		self.host = "127.0.0.1"

	def create_connection(self, port_number):

		channel = implementations.insecure_channel(self.host, int(port_number))
		stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
		return stub

	def predictResponse_into_nparray(self, response, output_tensor_name):
		dims = response.outputs[output_tensor_name].tensor_shape.dim
		shape = tuple(d.size for d in dims)
		if (output_tensor_name == 'detection_scores'):
			return_array = np.reshape(response.outputs[output_tensor_name].float_val, shape)
			# print(return_array*1)
		else:
			return_array = np.reshape(response.outputs[output_tensor_name].float_val, shape)
		return return_array

	def create_bb(self,detected_box,derived_scores,detection_classes,image_abs_path):
		
		img	   = cv2.imread(image_abs_path)
		output_save_path = image_abs_path
		h, w, _   = img.shape
		for index,box in enumerate(detected_box):
			class_detected = detection_classes[index]
			score	  = derived_scores[index]
			if(class_detected!=1):
				continue
			if(score<0.5):
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
	
	def serving_model_prediction(self, stub, batch_of_img, model_name, input_tensor, output_tensor, signature):
		print('\t - Enter serving_model_prediction')
		print(f" > Serving main function: {model_name}  ")
		request = predict_pb2.PredictRequest()
		request.model_spec.name = model_name
		request.model_spec.signature_name = signature  # ..> tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
		print(batch_of_img.shape)
		request.inputs[input_tensor].CopyFrom(
			tf.make_tensor_proto(batch_of_img, dtype='uint8'))
		result = stub.Predict(request, 30)

		# print(result)
		detection_boxes = self.predictResponse_into_nparray(result, 'detection_boxes')
		detection_scores = self.predictResponse_into_nparray(result, 'detection_scores')
		detection_classes = self.predictResponse_into_nparray(result, 'detection_classes')

		return detection_boxes, detection_scores, detection_classes

	def post_procesing(self,detection_boxes,detection_scores,detection_classes,base_info):
		_ids,image_paths = base_info
		outputs_list      = []
		for index,_id in enumerate(_ids):
			image_path = image_paths[index]
			_id = _ids[index]
			detection_box = detection_boxes[index]
			detection_score= detection_scores[index]
			detection_class = detection_classes[index]
			output_save_path = self.create_bb(detection_box,detection_score,detection_class,image_path)
			outputs_list.append({
					"id": _id,
					"annotated_path":output_save_path,
					"image_path": image_path
				})
		return outputs_list
	
	def batch_prediction(self, stub, model_meta, preprocessed_img_list, preprocessed_img_id_list,
						 actual_image_path_list):
		_ids, images_path = preprocessed_img_id_list, actual_image_path_list
		batch_output = {}
		model2run = model_meta['model_name']
		# time.sleep(5),
		print('\t - Entering batch_prediction')
		# input_tensor,output_tensor,signature,model_name,labels_list,model_type,width,height = extract_model_data(model_meta)

		input_tensor = model_meta['input_tensor']
		output_tensor = model_meta['output_tensor']
		signature = model_meta['signature']
		model_name = model_meta['model_name']
		model_platform = model_meta['model_platform']
		model_type = model_meta['model_type']

		batch_np = np.array(preprocessed_img_list)
		# time.sleep(5)
		detection_boxes, detection_scores, detection_classes = self.serving_model_prediction(stub=stub,
				 batch_of_img=batch_np,
				 model_name=model_name,
				 input_tensor=input_tensor,
				 output_tensor=output_tensor,
				 signature="serving_default")
		base_info = (preprocessed_img_id_list,actual_image_path_list)
		outputs_list = self.post_procesing(detection_boxes,detection_scores,detection_classes,base_info)
		return outputs_list

	def colonel_serving(self, model_meta, preprocessed_img_list, preprocessed_img_id_list, actual_image_path_list):
		# time.sleep(5)
		print('\t - Entering colonel batch')
		port_number = model_meta['port']
		stub = self.create_connection(port_number)
		batch_output = self.batch_prediction(stub, model_meta, preprocessed_img_list, preprocessed_img_id_list,
											 actual_image_path_list)
		del stub
		return batch_output


	def main(self, model2run, batch_of_image):
		result = {}
		preprocessed_img_list = []
		preprocessed_img_id_list = []
		actual_image_path_list = []
		nf_obj_meta = {
			"human_detection": {
				'input_tensor': 'inputs',
				'output_tensor': 'detection_scores',
				'signature': 'default',
				'model_name': 'human_detection',
				'model_platform': 'tensorflow',
				'model_type': 'keras',
				'port': '8001',
			}
		}
		meta_data_nf2 = nf_obj_meta[model2run]

		for single_image in batch_of_image:
			_id = single_image["id"]
			image_path = single_image["image_path"]
			image = load_img(image_path, target_size=(1024, 800))
			image = img_to_array(image)
			actual_image_path_list.append(image_path)
			preprocessed_img_list.append(image)
			preprocessed_img_id_list.append(_id)
		result = self.colonel_serving(meta_data_nf2, preprocessed_img_list, preprocessed_img_id_list,
									  actual_image_path_list)
		return result
