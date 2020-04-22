import json
import time
from typing import Dict, List

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cfunctions import *
from inference_client import InferenceHumanIdentifier
from serving_client import ServingHumanIdentifier

global serv_obj,inf_obj

serv_obj = ServingHumanIdentifier()
# inf_obj  = InferenceHumanIdentifier()

app = FastAPI()

class NestedInput(BaseModel):
    image: str
    id: str

class Inputs(BaseModel):
    input: List[NestedInput]


@app.post("/request")
async def ObjectDetectionPost(*, body: Inputs):
	body = dict(body)
	image_base64_list = body.get("input")
	if(len(image_base64_list)>16):
			return JSONResponse(
			status_code=status.HTTP_406_NOT_ACCEPTABLE,
			content=jsonable_encoder({"message": "Maximum limit for batch processing is 16 images in single req."}),
		)

	model_input,error_input = validate_base64(body["input"])
	if(len(model_input)==0):
		return JSONResponse(
			status_code=status.HTTP_400_BAD_REQUEST,
			content=jsonable_encoder({"message": "Inputs Batch must have atleast single valid request, valid image: JPEG or PNG (< 5MB).", "error_ids":error_input}),
		)
	try:
		model_output	        = serv_obj.main("human_detection",model_input)
		# model_output            = inf_obj.main(model_input)
		return_output           = post_processing(model_output)
	except:
		return JSONResponse(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			content=jsonable_encoder({"message": "Internal Server Error. TF-Serving corrupted or Switched OFF."}),
		)
	if(len(error_input)>0):
		error_dict = {
			"message": "Valid Image: JPEG or PNG (<5MB).",
			"error_ids": error_input
			}
	else:
		error_dict = {"error_ids": [None]}
	return {"output": return_output, "errors": error_dict}
