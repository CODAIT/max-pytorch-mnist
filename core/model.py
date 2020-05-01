#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from maxfw.model import MAXModelWrapper

import io
import logging
from PIL import Image
import torch
from torchvision import transforms
from config import DEFAULT_MODEL_PATH
from model import MyConvNet

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': 'minst-classifier',
        'name': 'MINST Classifier',
        'description': 'A simple image classifier for MINST.',
        'type': 'Image Classifier',
        'source': '',
        'license': 'My Open Source License'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))
        self.net = MyConvNet()
        self.net.load_state_dict(torch.load(path))
        logger.info('Loaded model')

        # Transform like what the training has done
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def _pre_process(self, inp):
        with Image.open(io.BytesIO(inp)) as img:
            img = self.transform(img)
            img = img[None, :, :]  # Create a batch size of 1
            logger.info('Loaded image... %d', img.size)
            return img

        return None

    def _post_process(self, result):
        probability, prediction = torch.max(result, dim=1)
        return [{'probability': probability,
                 'prediction': prediction}]

    def _predict(self, x):
        return self.net(x)
