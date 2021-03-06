from handy.py_rotate.rotate import rotate
from meta import AbstractModel


class Hybrid(AbstractModel):

    def __init__(self, dipole_inv_model, deblurring_model, is_joint=True, mode='train'):
        keys = ('pure_phi', 'angled_phi', 'rot', 'inv_rot', 'mask')
        super(Hybrid, self).__init__(keys, mode)
        self.model1 = dipole_inv_model
        self.model2 = deblurring_model
        self.joint_training = is_joint

    def train_model(self, pure_phi, angled_phi, rot, inv_rot, mask):
        pred1 = self.model1(pure_phi * mask)
        pred2 = rotate(self.model1(rotate(angled_phi, rot)), inv_rot) * mask
        pred3 = self.model2(pred2) * mask if self.joint_training else self.model2(pred2.detach())
        return pred1, pred2, pred3

    def eval_model(self, *args):
        return self.train_model(*args)[2]