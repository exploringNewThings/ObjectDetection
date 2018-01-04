from .LossBase import LossBase


class BaseLossBbox(LossBase):
    def __init__(self):
        super(BaseLossBbox,self).__init__("Base Loss Bbox")

    def loss(self, features, bboxes):
        pass
