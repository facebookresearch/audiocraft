import torch
import numpy as np
import typing as tp
# import ot

from ..quantization import gauss

def exp_kernel(dxx, a):
    return torch.exp((-0.5 / a) * dxx).sum()

def sliverman_rule(B, N):
    return B**(-1/(N+4))

def pairwise_distance_matrix(x: torch.Tensor, y: torch.Tensor, normalize: bool) -> torch.Tensor:
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxy = rx.t() + ry - 2. * xy # Used for C in (1)
    if normalize:
        dxy = dxy / torch.sum(dxy)
    return dxy

def batch_shuffling(x, groups: int = 0):
    ''' x is B,N
    If groups > 0 we do group shuffling i.e. we only permute between groups.
    The shuffling is the same intra group, so that only the groups
    will be independent between them, and we keep the intra-group correlation
    '''
    B, N = x.size()
    x_shuffled = torch.zeros_like(x)
    groups = groups if groups else N #That way, we keep the original version compatible

    assert N%groups == 0, f"Dimensions {N} are not divisible by number of groups {groups}"
    for group in range(groups):
        batch_perm = np.random.choice(np.arange(x.size(0)), replace=False, size=B)
        x_shuffled[:, group*N//groups: (group+1)*N//groups] = x[batch_perm, group*N//groups: (group+1)*N//groups]
    return x_shuffled

def _get_marginal_hist_bits(inputs, bits, marginal_hist):
    N = inputs.size(-1)
    inputs = inputs.cpu()
    for n in range(N):
        marginal_hist[n].scatter_add_(0, inputs[..., n], torch.ones_like(inputs[..., n], dtype=torch.float))
    return marginal_hist

def _get_joint_hist_bits(inputs, bits, joint_hist):
    N = inputs.size(-1)
    inputs = inputs.cpu()
    joint_inputs = inputs[..., 0]
    for n in range(1, N):
        joint_inputs = inputs[..., n] + 2**bits * joint_inputs
    joint_hist.view(-1).scatter_add_(0, joint_inputs, torch.ones_like(joint_inputs, dtype=torch.float))
    return joint_hist

def get_histograms_from_levels(inputs: torch.Tensor, bits: int, subsample: int, num_tuples: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    '''
        inputs: B*T, N (actually ..., N : all the first dimensions are considered as batch dimensions and subsequently flattened)
    '''
    try:
        assert inputs.dtype == torch.int64, "Levels was not typed as int"
    except AssertionError:
        assert torch.nn.MSELoss()(inputs.type(torch.int64), inputs) < 1e-8, "Levels is not an int representation at all"
        inputs = inputs.type(torch.int64)

    with torch.no_grad():

        if subsample:
            N = inputs.size(-1)
            assert subsample > 1 and subsample <= N, f"Wrong subsampling, {subsample} should be > 1 and <= {N}"
            subsample_dimensions = [ np.random.choice(np.arange(N), subsample, replace=False) for _ in range(num_tuples) ]
            marginal_histograms = torch.zeros((len(subsample_dimensions), subsample, 2**bits), device='cpu')
            joint_histogram = torch.zeros((len(subsample_dimensions), *subsample*(2**bits,)), device='cpu')

            for i_tuple, tuple_dimensions in enumerate(subsample_dimensions):
                marginal_histograms[i_tuple] = _get_marginal_hist_bits(inputs[..., tuple_dimensions], bits=bits, marginal_hist=marginal_histograms[i_tuple])
                joint_histogram[i_tuple] = _get_joint_hist_bits(inputs[..., tuple_dimensions], bits=bits, joint_hist=joint_histogram[i_tuple])

        else: #subsample_dimensions were already subsampled and input is num_tuples x n_points x subsample
            assert inputs.ndim == 3, f"Unexpected number of dimensions: expected num_tuples X n_points X subsample and got: {inputs.size()}"
            num_tuples, n_points, subsample = inputs.size()
            marginal_histograms = torch.zeros((num_tuples, subsample, 2**bits), device='cpu')
            joint_histogram = torch.zeros((num_tuples, *subsample*(2**bits,)), device='cpu')
            for i_tuple in range(num_tuples):
                marginal_histograms[i_tuple] = _get_marginal_hist_bits(inputs[i_tuple], bits=bits, marginal_hist=marginal_histograms[i_tuple])
                joint_histogram[i_tuple] = _get_joint_hist_bits(inputs[i_tuple], bits=bits, joint_hist=joint_histogram[i_tuple])
    return marginal_histograms, joint_histogram



# --------------- #



class Penalty(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, **kwargs):
        raise NotImplementedError

class NonePenalty(Penalty):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, **kwargs):
        return torch.zeros(1).requires_grad_(False), None, None

class DummyPenalty(Penalty):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, **kwargs):
        # distance = torch.nn.MSELoss()(inputs, torch.zeros_like(inputs))
        distance = torch.nn.MSELoss()(inputs, 3 + torch.randn_like(inputs)) ## Simply introduce noise
        return distance, None, None

class MonomesPenalty(Penalty):

    def __init__(self, max_order: int = 3, std_out: bool = True, **kwargs):
        super().__init__()
        self.max_order = max_order
        self.std_out = std_out

    def forward(self, inputs: torch.Tensor, **kwargs):
        '''inputs: B, T, N
        Compute a monome loss to enforce independence:
        Completely unstable

            L = E[ prod_k(X_k**d_k) ] - prod_k( E[ X_k**d_k ] )
        '''
        x = inputs.type(torch.float)
        x = (x - x.mean(0)) / x.std(0)

        penalty_list = []
        n_monomes = 10
        for _ in range(n_monomes):
            orders = torch.randint(self.max_order+1, (1, x.size(-1)))
            monomes = torch.pow(x, orders.to(x.device))
            std_monomes = monomes.clone()
            if self.std_out:
                for i in range(inputs.size(-1)):
                    if orders[0, i]: # Avoid having zeros everywhere when monome is constant
                        std_monomes[..., i] = (std_monomes[..., i] - monomes.mean(0)[..., i]) / (monomes.std(0)[..., i] + 1e-8)
                monomes = std_monomes

            m_marginals = monomes.mean(0)
            prod_marginals = torch.prod(m_marginals, dim=-1)

            m_joint = torch.prod(monomes, dim=-1)
            prod_joint = m_joint.mean(0)

            penalty_list.append(torch.nn.MSELoss()(prod_marginals, prod_joint))

        return torch.mean(torch.stack(penalty_list)), None, None

class RandomMLPsPenalty(Penalty):

    def __init__(self, std_in: bool = True, std_out: bool = True, **kwargs):
        super().__init__()
        self.std_in = std_in
        self.std_out = std_out

    def forward(self, inputs: torch.Tensor, **kwargs):
        '''inputs: B, T, N
        Compute a loss to enforce independence given some random MLPs:
            L = E[ prod_k( f_k(X_k) ) ] - prod_k( E[ f_k(X_k) ] )
        '''
        x = inputs.type(torch.float)
        x = (x - x.mean(0)) / x.std(0)

        penalty_list = []
        n_functions = 10
        for _ in range(n_functions):
            functions_independence = [torch.nn.Sequential(*[torch.nn.Linear(1, 20), torch.nn.ReLU(), torch.nn.LayerNorm(20), torch.nn.Linear(20, 20), torch.nn.ReLU(), torch.nn.LayerNorm(20), torch.nn.Linear(20, 1)]) for _ in range(inputs.size(-1))]
            for i in range(inputs.size(-1)):
                functions_independence[i].cuda()
                for param in functions_independence[i].parameters():
                    param.requires_grad = False

            transformed = torch.stack([ functions_independence[i](x[..., i].unsqueeze(-1)).squeeze(-1) for i in range(inputs.size(-1)) ], dim=-1)
            if self.std_out:
                transformed = (transformed - transformed.mean(0)) / (transformed.std(0) + 1e-8)

            m_marginals = transformed.mean(0)
            prod_marginals = torch.prod(m_marginals, dim=-1)

            m_joint = torch.prod(transformed, dim=-1)
            prod_joint = m_joint.mean(0)

            penalty_list.append(torch.nn.MSELoss()(prod_marginals, prod_joint))

        return torch.mean(torch.stack(penalty_list)), None, None

class MMDPenalty(Penalty):
    
    def __init__(self, kernel: str = 'rbf', device = None, time_steps: int = 0, 
                 delay_penalty: bool = False, group_norm_penalty: bool = False, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.device = device
        self.time_steps = time_steps
        self.delay_penalty = delay_penalty
        self.group_norm_penalty = group_norm_penalty

    def forward(self, inputs: torch.Tensor, groups: int = 0, 
                frames: int = 0, **kwargs):
        """Empirical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.
        when x = joint and y = factorized then it is a proxy for MI

        Args:
            # inputs: B,T,N
            inputs: B*T,N
            kernel: kernel type such as "multiscale" or "rbf
        """
        if self.device is not None:
            inputs = inputs.to(self.device)

        # Reshaping / Delaying
        x = inputs.type(torch.float) 
        if self.time_steps:
            x = x.reshape(-1, self.time_steps*x.size(-1)) #B*T/t_groups, t_groups*N
        elif self.delay_penalty:
            x = x.reshape(-1, frames, x.size(-1))
            x = torch.cat([ torch.nn.functional.pad(x[:, : frames-delay, delay*x.size(-1)//groups: (delay+1)*x.size(-1)//groups ], (0, 0, delay, 0)) for delay in range(groups) ], dim=-1)
            x = x[:, groups: ] #Crop to remove zeros introduced by padding, otherwise it messes up everything
            x = x.reshape(-1, inputs.size(-1))

        # Normalization
        if self.group_norm_penalty and groups:
            x = x.reshape(x.size(0), -1, groups)
            x = (x - x.mean((0, 1))) / torch.sqrt(x.var((0, 1)) + 1e-8)
            x = x.reshape(x.size(0), -1)
        else:
            x = (x - x.mean(0)) / torch.sqrt(x.var(0) + 1e-8)

        y = batch_shuffling(x, groups=groups)
        B = x.size(0)

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)

        out = 0.

        if self.kernel == "rbf":
            bandwidth_range = [0.1, 1, 5, 10, 20, 50]
            for a in bandwidth_range:
                out += (torch.utils.checkpoint.checkpoint(exp_kernel, dxx, a) - B) / (B * (B - 1))
                out += - 2 * (torch.utils.checkpoint.checkpoint(exp_kernel, dxy, a)) / B**2
                out += (torch.utils.checkpoint.checkpoint(exp_kernel, dyy, a) - B) / (B * (B - 1))

        elif self.kernel == "gaussian":
            a = (1.5 * sliverman_rule(*x.size()))**2
            out += (torch.utils.checkpoint.checkpoint(exp_kernel, dxx, a) - B) / (B * (B - 1))
            out += - 2 * (torch.utils.checkpoint.checkpoint(exp_kernel, dxy, a)) / B**2
            out += (torch.utils.checkpoint.checkpoint(exp_kernel, dyy, a) - B) / (B * (B - 1))

        out = torch.nn.ReLU()(out)
        return out, None, None

class MIHistogramPenalty(Penalty):

    def __init__(self, bits: int, subsample: int = 2, num_tuples: int = 10, time_steps: int = 0, device: str = None, delay_penalty: bool = False, **kwargs):
        super().__init__()
        self.bits = bits
        self.subsample = subsample
        self.num_tuples = num_tuples
        self.time_steps = time_steps
        self.device = device
        self.delay_penalty = delay_penalty

    def forward(self, inputs: torch.Tensor, normalize: bool = True, epsilon: float = 1e-10, frames: int = 50, groups: int = 0, **kwargs):
        ''' inputs is B*T,N
            marginal_histograms: len(subsample_dimensions), N, 2**bits
            joint_histogram: len(subsample_dimensions), N*(2**bits,)
        '''

        # assert not self.training, "Not  to be used during training"

        if self.device is not None:
            inputs = inputs.to(self.device)
        if self.time_steps:
            inputs = inputs.reshape(-1, self.time_steps*inputs.size(-1)) #B*T/T_groups , t_groups*N
        elif self.delay_penalty:
            x = x.reshape(-1, frames, x.size(-1))
            x = torch.cat([ torch.nn.functional.pad(x[:, : frames-delay, delay*x.size(-1)//groups: (delay+1)*x.size(-1)//groups ], (0, 0, delay, 0)) for delay in range(groups) ], dim=-1)
            x = x[:, groups: ] #Crop to remove zeros introduced by padding, otherwise it messes up everything
            x = x.reshape(-1, inputs.size(-1))

        marginal_histograms, joint_histogram = get_histograms_from_levels(inputs, bits=self.bits, subsample=self.subsample, num_tuples=self.num_tuples)
        mutual_information_list = []
        H_marginals_list = []
        H_joint_list = []
        for i_tuple in range(marginal_histograms.size(0)):

            marginal_pdfs = marginal_histograms[i_tuple] / torch.sum(marginal_histograms[i_tuple], dim=-1).unsqueeze(-1)
            joint_pdf = joint_histogram[i_tuple] / torch.sum(joint_histogram[i_tuple])

            H_marginals = torch.sum( torch.stack([ -torch.sum(pdf * torch.log2(pdf + epsilon), dim=-1) for pdf in marginal_pdfs], dim=0), dim=0)
            H_joint = -torch.sum(joint_pdf * torch.log2(joint_pdf + epsilon))
            mutual_information = H_marginals - H_joint

            if normalize:
                mutual_information = mutual_information / H_marginals
            mutual_information_list.append(mutual_information)
            H_marginals_list.append(H_marginals)
            H_joint_list.append(H_joint)

        mutual_information_mean = torch.Tensor(mutual_information_list).mean()
        H_marginals_mean = torch.Tensor(H_marginals_list).mean()
        H_joint_mean = torch.Tensor(H_joint_list).mean()

        return mutual_information_mean, H_marginals_mean, H_joint_mean

class PCCPenalty(Penalty):
    
    def __init__(self, device = None, delay_penalty: bool = False, **kwargs):
        super().__init__()
        self.device = device
        self.delay_penalty = delay_penalty

    def forward(self, inputs: torch.Tensor, groups: int = 0, frames: int = 50, **kwargs):
        """Pearson Correlation Coefficient
        Args:
            inputs: B*T,N
        """

        # assert not self.training, "Not  to be used during training"

        if self.device is not None:
            inputs = inputs.to(self.device)
        x = inputs.type(torch.float) #Helps regularizing

        if self.delay_penalty:
            x = x.reshape(-1, frames, x.size(-1))
            x = torch.cat([ torch.nn.functional.pad(x[:, : frames-delay, delay*x.size(-1)//groups: (delay+1)*x.size(-1)//groups ], (0, 0, delay, 0)) for delay in range(groups) ], dim=-1)
            x = x[:, groups: ] #Crop to remove zeros introduced by padding, otherwise it messes up everything
            x = x.reshape(-1, inputs.size(-1))
        if groups:
            x = x.view(-1, groups) #Take groups of dimensions used in PQ

        out = torch.corrcoef(x.t())

        return out, None, None




def get_independence_penalty_cls(independence_penalty, **kwargs):

    if independence_penalty == 'none':
        return NonePenalty()
    if independence_penalty == 'dummy':
        return DummyPenalty()
    elif independence_penalty == 'mmd':
        return MMDPenalty(kernel='rbf', **kwargs)
    elif independence_penalty == 'mi-histo':
        return MIHistogramPenalty(**kwargs)
    elif independence_penalty == 'pcc':
        return PCCPenalty()


    elif independence_penalty == 'monome':
        raise NotImplementedError
        return MonomesPenalty(max_order=3, std_in=True, std_out=True)
    elif independence_penalty == 'mlp':
        raise NotImplementedError
        return RandomMLPsPenalty(std_in=True, std_out=True)
    elif independence_penalty == 'mi-renyi-alpha':
        raise NotImplementedError
        return MIRenyiAlphaPenalty(alpha=1.01)
    elif independence_penalty == 'ot':
        raise NotImplementedError
        return OTSoftTargetPenalty()
    elif independence_penalty == 'ot-dist':
        raise NotImplementedError
        return OTDistancePenalty()
