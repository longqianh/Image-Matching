import cv2
from pathlib import Path
import torch
import numpy as np
from models.matching import Matching
from models.utils import frame2tensor
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose,
                          pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


class Arg(object):

  def __init__(self,
               config={},
               _input='./assets/input_img/',
               _output=None,
               scene_mode='outdoor',
               skip=1,
               img_glob=['*.png', '*.jpg', '*.jpeg'],
               max_length=1000000):

    super(Arg, self).__init__()
    # self.nms_radius=nms_radius
    # self.keypoint_threshold=keypoint_threshold
    self.input_dir = _input
    # 'ID of a USB webcam, URL of an IP camera, '
    # 'or path to an image directory or movie file'
    self.config = config
    self.output_dir = _output
    if scene_mode == 'outdoor':
      self.resize = [1600]
      self.max_kpts = 2048
      self.nms_radius = 3
    else:
      self.resize = [640]
      self.max_kpts = 1024
      self.nms_radius = 4

    self.skip = skip
    self.img_glob = img_glob
    self.max_length = max_length

  def set_config():
    pass

  def set_anchor_img():
    pass


def get_similarity(img1, img2, args, device='cpu'):
  anchor = frame2tensor(img1, device)
  to_match = frame2tensor(img2, device)

  matching = Matching(args.config).eval().to(device)  # matching model
  pred = matching({'image0': anchor, 'image1': to_match})
  conf = pred['matching_scores0']
  return torch.mean(conf).item(), torch.max(conf).item()


# 下一步：set anchor以后想办法加速 把kpts直接存好 应该可以快一倍

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
torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
args = Arg(scene_mode='outdoor')
name0 = 'anchor.jpg'
name1 = 'tomatch.jpg'
# image0, inp0, scales0 = read_image(
#     args.input_dir + name0, device, args.resize)
# image1, inp1, scales1 = read_image(
#     args.input_dir + name1, device, args.resize)
# matches_path = 'assets/match_res.npz'
# print(inp0)
img0 = cv2.imread(args.input_dir + name0, cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(args.input_dir + name1, cv2.IMREAD_GRAYSCALE)
# print(get_similarity(img0, img1, args, device))
