import torch
from einops import rearrange


def _exp_kernel(dxx: torch.Tensor, a: torch.Tensor):
    return torch.exp((-0.5 / a) * dxx).sum()


def _shuffle_codebooks(x):
    ''' x is B,K,D
    '''
    B, K, D = x.size()
    x_shuffled = torch.zeros_like(x)

    for k in range(K):
        batch_perm = torch.randperm(B, device=x.device)
        x_shuffled[:, k, :] = x[batch_perm, k, :]
    return x_shuffled


class MMDLoss(torch.nn.Module):
    def __init__(self, delay: bool = False, device=None, group_norm_mmd: bool = False):
        super().__init__()
        self.device = device
        self.delay = delay
        self.group_norm_mmd = group_norm_mmd

    def forward(self, inputs: torch.Tensor):
        """inputs is [B, K, D, T].
        """
        if self.device is not None:
            inputs = inputs.to(self.device)

        B, K, D, T = inputs.size()
        x = inputs.type(torch.float)

        # Reshaping / Delaying
        if self.delay:
            x = torch.cat([
                torch.nn.functional.pad(x[:, delay: (delay+1), :, : T-delay], (delay, 0)) for delay in range(K)
                ], dim=1)
            x = x[..., K:]  # Crop to remove zeros introduced by padding

        # Group time dimension and shuffle to sample from factorized distribution
        x = rearrange(x, 'b k d t -> (b t) k d')
        # Normalize per batch example
        x = (x - x.mean(dim=(0,), keepdim=True)) / torch.sqrt(x.var(dim=(0,), keepdim=True) + 1e-8)
        y = _shuffle_codebooks(x)
        x = x.view(x.shape[0], -1)
        y = y.view(x.shape[0], -1)

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz

        out: torch.Tensor = 0.0  # type: ignore

        bandwidth_range = [0.1, 1, 5, 10, 20, 50]
        for a in bandwidth_range:
            out += (torch.utils.checkpoint.checkpoint(_exp_kernel, dxx, a) - B) / (B * (B - 1))
            out += (-2 / B**2) * (torch.utils.checkpoint.checkpoint(_exp_kernel, dxy, a))
            out += (torch.utils.checkpoint.checkpoint(_exp_kernel, dyy, a) - B) / (B * (B - 1))

        return out.clamp(min=0)
