

class DiceLoss(object):
    """docstring forDiceLoss."""

    def __init__(self):
        self.x= 0


    def Loss(self, unet, tensor):
        y_pred = unet.forward(tensor.tensorX)
