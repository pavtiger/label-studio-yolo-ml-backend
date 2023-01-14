from label_studio_ml.model import LabelStudioMLBase

import os
import cv2
import pathlib
import random

import sys
sys.path.insert(1, 'yolov7')

from models.experimental import attempt_load
from utils.torch_utils import select_device

from hubconf import custom


LABEL_STUDIO_DATA_PATH = "/home/pavtiger/Docs/label-studio/mydata/media"
yolov5_repo_dir = r"yolov7"
pretrained_model_path = r"yolo.pt"
model = custom(path_or_model=pretrained_model_path)


class DummyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(DummyModel, self).__init__(**kwargs)

        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: 
            model returns the list of predictions based on input list of tasks

            :param tasks: Label Studio tasks in JSON format
        """

        for i, task in enumerate(tasks):
            print(task['data']['image'])
            image_path = pathlib.Path(task['data']['image'])
            new_path = pathlib.Path(*image_path.parts[2:])

            img = cv2.imread(os.path.join(LABEL_STUDIO_DATA_PATH, new_path), cv2.COLOR_BGR2RGB)
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

            results = model(imgs)
            labels = results.xyxyn[0].cpu().numpy()
            names = model.names

            results.print()
            results.save()

            n = len(labels)
            x_shape, y_shape = img.shape[1], img.shape[0]

            results = []
            for elem in labels:
                row = elem[:-1]

                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 0, 255)
                print((x1, y1), (x2, y2))

                cls = int(elem[-1])

                result = {
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": x_shape,
                    "original_height": y_shape,
                    "image_rotation": 0,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [str(names[cls])],
                        'x': elem[0] * 100,
                        'y': elem[1] * 100,
                        'width': (elem[2] - elem[0]) * 100,
                        'height': (elem[3] - elem[1]) * 100
                    },
                    'score': float(elem[-2]),
                    'model_version': 'test123'
                }
                results.append(result)

        return [{
            'result': results,
            'score': float(0.4)
        }]


    def fit(self, completions, workdir=None, **kwargs):
        """ This is where training happens: train your model given list of completions,
            then returns dict with created links and resources
            :param completions: aka annotations, the labeling results from Label Studio 
            :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result
        pass
        return {}
