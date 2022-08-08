import random
import torchvision.transforms.functional as f

class Randomrotate(object):
    def __init__(self,angles,seed=1):
        self.angles = (-angles,angles)
        random.seed(seed)
    @staticmethod
    def get_params(angles):
        angle = random.uniform(angles[0], angles[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.angles)
        return f.rotate(img, angle, False, False, None, None)