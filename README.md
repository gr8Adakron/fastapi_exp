# Human detection model deployment using FastAPI:


Deployment perform using two different channels: **Tensorflow Serving and Old School Direct Inference**

## Installations:

**Mandatory:** Python3.6 </br>
**Two Assumptions:**  
* Already created an seprate python environment for this repo.
* Cloning this repo in your Systems $HOME folder as for the further coming commands.

```
#..> Base Installation
cd $HOME
git clone git@github.com:gr8Adakron/fastapi_exp.git
cd $HOME/fastapi_exp
pip install -r statics/fastapi.env

#..> Installing TF objection package
git clone --depth 1 https://github.com/tensorflow/models
cd $HOME/fastapi_exp/models/research
pip install .

#..> TF-Serving Docker Installation.
#..> CPU Version.
sudo docker pull tensorflow/serving

#..> Incase for GPU Version 
sudo docker pull tensorflow/serving:latest-gpu

```

## API Initiation Commands:

* You need to open 2-terminals inorder to make it work.
* **Terminal #1:** For starting FastAPI server for accepting request.
* **Terminal #2:** For starting TF-Serving docker image in order to load the models efficiently.

#### Terminal #1 Commands (FAST API SERVER):
```
cd $HOME/fastapi_exp
CORES="$(nproc --all)" && uvicorn main:app --workers ${CORES} --host 0.0.0.0 --port 8000

```

#### Terminal #2 Commands (TF Serving SERVER):
```
cd $HOME/fastapi_exp
CORES="$(nproc --all)" && sudo docker run -p 8001:8500 -p 8002:8501   --mount type=bind,source=/home/$USER/fastapi_exp/,target=/models/my_model   --mount type=bind,source=/home/$USER/fastapi_exp/models.config,target=/models/models.config   -t tensorflow/serving --model_config_file=/models/models.config --tensorflow_session_parallelism=${CORES} --enable_batching=true --enable_model_warmup=true 

```
Inorder to stop the docker use following command `sudo docker container stop $(sudo docker container ls -aq)`

**Note:** Both the commands considers the parallelism based on the number of cores available in the Systems. You can alter it as per the needs.


## Video Feeding To The API:
* Sample video of 30 sec present in the repo.
* Here we got two possible ways (scripts):
  1) By streaming frame by frame sequentially- which is little slow based on the system been hosted **i.e: [base_video_feed.py](base_video_feed.py)**
  2) By Feeding Concurrently along with batching.
    **i.e: [async_video_feed.py](async_video_feed.py)**

####  Scripts calling commands (Terminal #3):
```
#..> Assuming you will same environment for the feeding too.


#..> For streaming:
python  base_video_feed.py

#..> For Async:
python base_video_feed.py

```
##  Tweaking Variables As Per The Requirements:

1. Model can be called using TF-Serving Inference or Old School Inferencing:
    * TF-Serving Inference is Default in [main.py](main.py) as its efficient.
    * In order to call using Inferencing script, uncomment falling lines

```
#..> Uncomment Line number 18 in main.py.
18) inf_obj  = InferenceHumanIdentifier()

#..> Comment Line number 47 in main.py
47) model_output	        = serv_obj.main("human_detection",model_input)

#..> Uncomment Line number 18 in main.py
48) model_output            = inf_obj.main(model_input)

#..> resave it and restart Terminal#1.
#..> Turn off Terminal#2 as not needed.

```

2. Tweaking Async call and Batch size in video feeding scripts [async_video_feed.py](async_video_feed.py)

```
#..> Line number 147 and 148
batch_size          = 1
async_workers_count = 2

```

## API I/O Details:

**POST URL:** http://127.0.0.1:8000/request

**INPUT JSON:**
```
{
    "input":[
        {
            "image": "base64_string",
            "id": String
        }
    ]
}

```

**OUTPUT JSON:**
```
{
    "output":[
        {
            "annotated_image":"base64_string",
            "id": String
        }
    ]
    "errors": dict
}

```


