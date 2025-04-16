import torch
import torch.nn as nn

from pytorch_wavelets import DWT1DForward

class FKD(nn.Module):
    """Frequency distillation loss.

    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        level (int): Defaults to 3.
        basis (string): Defaults to 'db6'.
    """
    def __init__(
        self,
        loss_weight=1.0,
        level=3,
        basis='db6'
    ):
        super(FKD, self).__init__()

        self.xfm = DWT1DForward(J=level, wave=basis)
        self.loss_weight = loss_weight

    def forward(self, y_s, y_t):

        assert y_s.shape == y_t.shape
        loss = self.get_wavelet_loss(y_s, y_t)
        return self.loss_weight * loss
    
    def get_wavelet_loss(self, student, teacher):
        '''
        s: B, C, L
        t: B, C, L
        '''
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)

        loss = 0.0
        for index in range(len(student_h)):
            loss+= torch.nn.functional.l1_loss(teacher_h[index], student_h[index])
        return loss
 

class SKD(nn.Module):
    """Spatial distillation loss.

    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0):
        super(SKD, self).__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        assert y_s.shape == y_t.shape
        
        loss = self.mse(y_s, y_t)
        return self.loss_weight * loss


class QueryKD(nn.Module):
    """Query distillation loss.

    Inputs:
        s: B, L, C
        t: B, L, C
    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """
    def __init__(self, stu_channel, tea_channel, loss_weight=1.0):
        super(QueryKD, self).__init__()
        self.loss_weight = loss_weight
        self.skd = SKD(loss_weight=1.0)
        self.fkd = FKD(loss_weight=1.0)

        self.align_module = nn.Conv1d(stu_channel, tea_channel, \
                            kernel_size=1)

    def forward(self, y_s, y_t):
        y_s = y_s.transpose(1, 2) # B, C, L
        y_t = y_t.transpose(1, 2)

        y_s = self.align_module(y_s)
        
        assert y_s.shape == y_t.shape
        
        loss = self.skd(y_s, y_t) + self.fkd(y_s, y_t)
        return self.loss_weight * loss


if __name__ == '__main__':
    # test
    feat = torch.randn(1, 32, 768)
    feat_t = torch.randn(1, 32, 768)

    model = QueryKD(stu_channel=768, tea_channel=768)
    loss = model(feat, feat_t)
    print(loss)
