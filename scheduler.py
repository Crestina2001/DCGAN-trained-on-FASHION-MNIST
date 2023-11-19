import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class GapAwareScheduler(_LRScheduler):
    def __init__(self, optimizer, 
             V_star=math.log(4), 
             alpha=0.95, 
             hmin=0.1, 
             fmax=2, 
             xmin=0.1 * math.log(4), 
             xmax=0.1 * math.log(4), 
             last_epoch=-1):
        self.V_star = V_star
        self.alpha = alpha
        self.hmin = hmin
        self.fmax = fmax
        self.xmin = xmin
        self.xmax = xmax
        self.Vd_hat = V_star  # Initialize moving average of loss with ideal loss
        super(GapAwareScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Assuming a single parameter group for simplicity
        current_lr = self.optimizer.param_groups[0]['lr']
        gap = self.Vd_hat - self.V_star

        if gap >= 0:
            factor = min(self.fmax, 1 + gap / self.xmax)
        else:
            factor = max(self.hmin, 1 - abs(gap) / self.xmin)

        return [current_lr * factor]

    def update_loss(self, current_loss):
        self.Vd_hat = self.alpha * self.Vd_hat + (1 - self.alpha) * current_loss
