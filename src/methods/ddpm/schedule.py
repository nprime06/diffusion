import torch

class DDPMScheduler: # everything is on cpu
    def __init__(self, beta_start, beta_end, num_steps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        self.alphas = (1.0 - self.betas).clamp(min=1e-12)
        self.log_alpha_cumprod = torch.cumsum(torch.log(self.alphas), dim=0)
        self.alpha_cumprod = torch.exp(self.log_alpha_cumprod)

    def _index(self, t: torch.Tensor) -> torch.Tensor:  
        return (t - 1).long().detach().cpu()

    def get_beta(self, t): # t is 1 indexed
        idx = self._index(t)
        return self.betas[idx].float()

    def get_alpha(self, t):
        idx = self._index(t)
        return self.alphas[idx].float()

    def get_alpha_cumprod(self, t):
        idx = self._index(t)
        return self.alpha_cumprod[idx].float()

    def get_standardized_time(self, t): # t is (B,)
        return (t.float() - 1.0) / (self.num_steps - 1)