import torch.nn as nn
import torch


class HourlySwingCertitudeAwareLoss(nn.Module):
    def __init__(self, punish_cert=0.2):
        super(HourlySwingCertitudeAwareLoss, self).__init__()
        self.punish_cert = punish_cert

    def forward(self, prediction, certitude, label):
        # prediction, certitude -> BS
        # label -> BS * 2 : 0 for direction, 1 for value
        # direction is 1 for swing high and -1 for swing low
        half_band = certitude / 2
        mid_target = label[:, 1] - (label[:, 0] * half_band)
        sq_error = (mid_target - prediction)**2
        sq_error = sq_error[sq_error > (half_band**2)]
        mse = torch.mean(sq_error) if sq_error.shape[0] > 0 else 0

        cert_err = self.punish_cert * torch.mean(certitude)

        return mse + cert_err
