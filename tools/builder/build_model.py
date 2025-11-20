# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from models.builder import build_detector


def build_model(cfg, train_cfg=None, test_cfg=None):
    build_detector(cfg, train_cfg, test_cfg)
