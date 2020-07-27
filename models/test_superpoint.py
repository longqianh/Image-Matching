#  Origin:
#  Magic Leap, Inc.
#  Unpublished Copyright (c) 2020
#  Originating Authors: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz
#  delete and fine tune: Peter_H

from pathlib import Path
import torch
from torch import nn


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    # 快速非极大值抑制（Non-maximum suppression，NMS）算法
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    # print(max_pool(scores))
    # print(max_mask)
    for _ in range(2):
        # print(max_mask.float())
        supp_mask = max_pool(max_mask.float()) > 0
        # print(supp_mask)
        supp_scores = torch.where(supp_mask, zeros, scores)  # True不变，False清零
        # print(supp_scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        # print(new_max_mask)
        max_mask = max_mask | (new_max_mask & (~supp_mask))  # 这是什么神仙操作。。
    return torch.where(max_mask, scores, zeros)

# test for simple_nmc func
# torch.manual_seed(4)
# t = torch.randn(1, 1, 4, 4)
# print(t)
# print(simple_nms(t, 1))


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (
        keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w

    # test for mask : False will be discarded
    # print(scores.shape)
    # mask[0] = False
    # print(scores[mask].shape)

    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """

    # print(descriptors.shape) #torch.Size([1, 256, 60, 80])
    b, c, h, w = descriptors.shape
    # print(keypoints.shape)
    # 下面两句是在干什么..
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    # print(keypoints) # to(keypoints) 指定dtype
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    # print(keypoints.view(b, 1, -1, 2).shape) # 插入一维,且控制最后一维是2
    # grid_sample 是将一个source_image，
    # 通过双线性插值的方式变换到另一个大小指定的target_image中
    # 这里 [1,256,60,80]-->[1,256,1,978]
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    # print(descriptors.shape) #torch.Size([1, 256, 978])

    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)  # inplace=True: 覆盖原来的
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'

        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):

        # now we look into the conv and see how the image change
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))

        x = self.relu(self.conv1b(x))

        # print(x.shape)
        x = self.pool(x)

        # print(x.shape)
        x = self.relu(self.conv2a(x))

        # print(x.shape)
        x = self.relu(self.conv2b(x))

        # print(x.shape)
        x = self.pool(x)

        # print(x.shape)
        x = self.relu(self.conv3a(x))

        # print(x.shape)
        x = self.relu(self.conv3b(x))

        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv4a(x))
        # print(x.shape)
        x = self.relu(self.conv4b(x))
        # print(x.shape)

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))

        scores = self.convPb(cPa)
        # print(scores.shape)
        scores = torch.nn.functional.softmax(
            scores, 1)[:, :-1]  # 沿着dim=1对65个矩阵进行softmax [:-1] 抽走最后一张
        # print(scores)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(
            b, h, w, 8, 8)  # 合并 每8个合成一个
        # print(scores)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * 8, w * 8)  # 这是什么神仙操作。。最后又组成了一个和原图大小一样的图片
        scores = simple_nms(scores, self.config['nms_radius'])

        # img = scores.detach().numpy()[0]
        # cv2.imshow('', img * 255)
        # cv2.waitKey(0)
        keypoints = [
            torch.nonzero(
                s > self.config['keypoint_threshold'], as_tuple=False)
            for s in scores]
        # 得分超过阈值的点是特征点，nonzero得到它们的坐标
        # print(keypoints[0])  # torch.Size([978, 2])
        # print(scores.shape)

        # print(s[tuple(k.t())])

        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # print(scores[0])
        # print(scores[0].shape)
        #?
        # tmp = []
        # for s, k in zip(scores, keypoints):
        #     # print(s[tuple(k.t())])
        #     # tmp.append(s[tuple(k.t()])
        #     print(s.shape)
        #     print(s[k.t()])
        #     print(k.t())
        #     print(tuple(k.t()))
        # tensor.t() 转置

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        # zip(*) [(1, 4), (2, 5), (3, 6)]-->[(1, 2, 3), (4, 5, 6)]

        # Convert (h, w) to (x, y) ?
        # []消去一维，然后再把里面的安装dim=1 (列) 翻转
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        # print(keypoints[0])

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(
            descriptors, p=2, dim=1)  # 除以最大Lp范数

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        # y = descriptors[0].detach().numpy()
        # print(y.shape)

        return {
            'scores': scores,  # feature points' score
            'keypoints': keypoints,  # keypoints' coordinate
            'descriptors': descriptors,  # feature points' descriptors
        }


if __name__ == '__main__':

    import cv2
    from utils import (compute_pose_error, compute_epipolar_error,
                       estimate_pose,
                       pose_auc, read_image,
                       rotate_intrinsics, rotate_pose_inplane,
                       scale_intrinsics,
                       plot_image_pair)
    impath0 = '../assets/input_img/anchor1.jpg'
    impath1 = '../assets/input_img/to_match.jpg'
    img0, img_tensor0, _ = read_image(impath0)
    img1, img_tensor1, _ = read_image(impath1)
    # plot_image_pair([img0, img1])
    m = SuperPoint({})
    data = {"image": img_tensor0}
    res = m(data)
    print(res['descriptors'][0].shape)  # torch.Size([256, 691]) 每一列是一个特征描述向量?
    print(res['scores'][0].shape)  # torch.Size([691])
    print(res['keypoints'][0].shape)  # torch.Size([691, 2])
