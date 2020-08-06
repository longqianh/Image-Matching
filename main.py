import cv2
from pathlib import Path
import torch
import numpy as np
from models.matching import Matching
from models.superpoint import SuperPoint
from models.utils import (read_image,frame2tensor,pt2onnx,onnx2pb,do_transfer)


class Arg(object):

  def __init__(self,
               _input='./assets/input_img/',
               _output=None,
               scene_mode='outdoor',
               skip=1,
               img_glob=['*.png', '*.jpg', '*.jpeg'],
               max_length=1000000):

    super(Arg, self).__init__()
    self.set_config()
    self.input_dir = _input
    self.output_dir = _output

    self.skip = skip
    self.img_glob = img_glob
    self.max_length = max_length

    self.data = {}
    self.superpoint = SuperPoint(self.config.get('superpoint', {}))

  def set_input_dir(self, input_dir):
    self.input_dir = input_dir

  def set_output_dir(self, output_dir):
    self.output_dir = output_dir

  def set_skip(self, skip):
    self.skip = skip

  def set_max_length(self, max_length):
    self.max_length = max_length

  def set_img_glob(self, glob):
    self.img_glob = img_glob

  def set_config(self, config=None):
    if config == None:
      self.config = {'superpoint': {
          'nms_radius': 4,
          'keypoint_threshold': 0.005,
          'max_keypoints': -1
      },
          'superglue': {
          # choices={'indoor', 'outdoor'}
          'weights': 'outdoor',
          'sinkhorn_iterations': 20,
          'match_threshold': 0.2,
      }}  # default_config

    else:
      self.config = config

  def set_superopint(self, config):
    self.superpoint = SuperPoint(config.get('superpoint', {}))

  def set_scene_mode(self, mode):
    if mode == 'outdoor':
      self.resize = [1600]
      self.max_kpts = 2048
      self.nms_radius = 3
    elif mode == 'indoor':
      self.resize = [640]
      self.max_kpts = 1024
      self.nms_radius = 4

  def set_anchor(self, img_name):

    anchor_tensor = read_image(self.input_dir + img_name)
    self.data['image0'] = anchor_tensor

    anchor_feature = self.superpoint({'image': anchor_tensor})
    self.data = {**self.data, **
                 {k + '0': v for k, v in anchor_feature.items()}}

  def set_match(self, match_name):
    # put image to match into the data
    # 似乎更慢了
    match_tensor = read_image(self.input_dir + match_name)
    self.data['image1'] = match_tensor
    match_feature = self.superpoint({'image': match_tensor})
    if self.data == None:
      self.data = {k + '1': v for k, v in match_feature.items()}
    else:
      self.data = {**self.data, **
                   {k + '1': v for k, v in match_feature.items()}}


def get_similarity(args, name, device='cpu'):
  args.set_match(name)
  matching = Matching(args.config).eval().to(device)  # matching model
  return matching(args.data)

if __name__ == '__main__':
  args = Arg(scene_mode='outdoor')
  args.set_anchor('zju1.jpg')
  args.set_match('zju2.jpg')
  # print(get_similarity(args, 'zju2.jpg'))
  m=Matching(args.config).eval()
  # ms=torch.jit.trace(m,args.data)
  # pt2onnx(m,args.data)
  print(m(args.data))



  # torch.set_grad_enabled(False)
  # device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  # name0 = 'anchor.jpg'
  # args.set_anchor(name0)
  # name1 = 'to_match.jpg'
  # name2 = 'to_match2.jpg'
  # name3 = 'to_match3.jpg'
  # name4 = 'to_match4.jpg'
  # print(get_similarity(args, name1))
  # print(get_similarity(args, name2))
  # # print(get_similarity(args, name3))
  # # print(get_similarity(args, name4))
