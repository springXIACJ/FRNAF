import torch

class NormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self,  eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x1, x2):
        _,c, h, w = x1.shape
        x1, x2 = self.norm(x1), self.norm(x2)
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= c * h * w
        return score

    def norm(self, x):
        mu = x.mean(dim=[-1, -2], keepdim=True)
        var = x.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
        std = var.sqrt()
        return (x - mu) / std
def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    #loss_mse = torch.mean((x-y)**2)
    loss_mse = torch.sum((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_ncc_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    #loss_mse = torch.mean((x-y)**2)
    loss_mse = torch.sum((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss




