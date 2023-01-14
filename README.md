# label-studio-yolo-ml-backend

Clone repostory with submodules
```shell
git clone git@github.com:pavtiger/label-studio-yolo-ml-backend.git --recursive
```
or just clone as usual and pull submodules with this command
```shell
git submodule update --init --recursive
```

## Installation
### It is suggested to use python `venv` for libraries installation
Activate venv
```shell
mkdir venv
python -m venv ./venv
source venv/bin/activate
```

Install requirements
```shell
pip install -r yolov7/requirements.txt  # yolo requirements
pip install -U -e label-studio-ml-backend  # install label studio backend
pip install redis rq  # additional libraries for the backend
```

## Running backend
```shell
label-studio-ml init minecraft_copilot --script ./main.py --force
label-studio-ml start minecraft_copilot
```
The ML backend server becomes available at http://localhost:9090

## Connecting to ML backend
Add an ML backend using the Label Studio UI

* In the Label Studio UI, open the project that you want to use with your ML backend.
* Click Settings > Machine Learning.
* Click Add Model.
* Type a Title for the model and provide the URL for the ML backend. For example, http://localhost:9090.
* (Optional) Type a description.
* (Optional) Select Use for interactive preannotation. See Get interactive pre-annotations for more.
* Click Validate and Save.

Instructions to connect taken from [label studio website](https://labelstud.io/guide/ml.html)
