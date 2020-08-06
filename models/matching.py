#  Origin:
#  Magic Leap, Inc.
#  Unpublished Copyright (c) 2020
#  Originating Authors: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz
#  delete and fine tune: Peter_H

import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue

# from superpoint import SuperPoint
# from superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        # config.get('superglue', {}):
        # if superglue not in the dict, return {}

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k + '0': v for k, v in pred0.items()}}

        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k + '1': v for k, v in pred1.items()}}

        # print(pred)
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = self.superglue(data)['matching_scores0']
        return torch.mean(pred).item()

        # pred = {**pred, **self.superglue(data)}

        # return pred


if __name__ == '__main__':
    from utils import read_image
    default_config = {'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
        'superglue': {
        # choices={'indoor', 'outdoor'}
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }}
    m = Matching(default_config).eval()  # matching model
    # name0 = 'anchor.jpg'
    # name1 = 'tomatch.jpg'
    # name2 = 'to_match2.jpg'
    # anchor_t = read_image("../assets/input_img/" + name0 )
    # print(anchor_t.shape)
    # ts=torch.jit.script(m)
    # print(m(anchor_t))
    # # superpoint = SuperPoint(default_config.get('superpoint', {}))

    # test1 = read_image(
    #     "../assets/input_img/" + name1)
    # test2 = read_image(
    #     "../assets/input_img/" + name2)
    # # print(timg0.shape)
    # data = {"image0": anchor_t, "image1": test1}
    # # print(m(data))  # 5.7s
    # s1 = m(data)

    # data["image1"] = test2
    # s2= m(data)
    # # print(m(data))
