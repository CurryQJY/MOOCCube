import torch


def sparsemax(input, dim=-1):
    z = input - input.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_shape = [1] * z.dim()
    range_shape[dim] = z.size(dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(range_shape)
    z_cumsum = zs.cumsum(dim)
    support = 1 + k * zs > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_sum = torch.gather(z_cumsum, dim, k_z - 1)
    tau = (tau_sum - 1) / k_z.to(dtype=z.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax15(input, dim=-1, n_iter=50):
    x = 0.5 * input
    x = x - x.max(dim=dim, keepdim=True).values
    tau_lo = x.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) * 0.5
        probs = torch.clamp(x - tau, min=0.0).pow(2)
        too_large = probs.sum(dim=dim, keepdim=True) >= 1.0
        tau_lo = torch.where(too_large, tau, tau_lo)
        tau_hi = torch.where(too_large, tau_hi, tau)
    tau = (tau_lo + tau_hi) * 0.5
    probs = torch.clamp(x - tau, min=0.0).pow(2)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
