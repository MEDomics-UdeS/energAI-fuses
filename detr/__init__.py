# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .criterion import build_criterion


def build(args):
    return build_criterion(args)
