#  Origin:
#  Magic Leap, Inc.
#  Unpublished Copyright (c) 2020
#  Originating Authors: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz
#  delete and finetuning: Peter_H

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread

import numpy as np
import cv2
import torch
import onnx
from onnx_tf.backend import prepare

# import matplotlib.pyplot as plt

# -- Image Processing --

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device='cpu'):
    # print(torch.from_numpy(frame/255.).float())
    # None 增加两个维度
    # print(torch.from_numpy(frame/255.).float()[None,None].shape)
    # torch.Size([1, 1, 480, 640])

    # cv2.imshow('1',frame)
    # cv2.waitKey(0)
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def read_image(path, device='cpu', resize=[640, 480], rotation=0, resize_float=0):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    # scales = (float(w_new) / float(w), float(h_new) / float(h))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    img_tensor = frame2tensor(image, device)
    # return image, img_tensor, scales
    return img_tensor


# -- model exporting --

def pt2onnx(model, dummy_input, model_path='./model.onnx'):
    torch.onnx.export(model, dummy_input, model_path, opset_version=12)


def onnx2pb(model, pb_output_path):
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


def do_transfer():
    pt_model_path = './pretrained-models/kinetics-joint.pt'
    onnx_input_path = './msg3d.onnx'
    pb_output_path = './msg3d.pb'

    pt_model = msg3d.Model(num_class=400, num_point=18, num_person=2,
                           num_gcn_scales=8, num_g3d_scales=8, graph='graph.kinetics.AdjMatrixGraph')
    pt_model.eval()
    state=pt_model.state_dict()
    pretrained_state_dict=torch.load(pt_model_path)
    # pt_model.load_state_dict(torch.load(pt_model_path))
    state.update(pretrained_state_dict)
    pt_model.load_state_dict(state)
    dummy_input = torch.randn(3, 50, 18, 2)
    pt2onnx(pt_model, dummy_input, onnx_input_path)

    # onnx_model = onnx.load(onnx_input_path)
    # onnx2pb(onnx_model, pb_output_path)

def create_graph(filename="model.pb"):
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

