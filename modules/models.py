import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.studentT import StudentT
import matplotlib.pyplot as plt


class ABC_time_model(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = torch.nn.Parameter(torch.zeros(5))
        else:
            self.theta = theta
        self.last_n_traj = None


    def forward(self, x, return_n=True):
        # x: [B, T]
        B, T = x.shape
        device = x.device
        dtype = x.dtype
        n = torch.zeros(B, device=device, dtype=dtype)
        n_traj = torch.empty(B, T, device=device, dtype=dtype) if return_n else None
        outputs = torch.empty(B, T, device=device, dtype=dtype)
        theta0, theta1, theta2, theta3, theta4 = self.theta[0], self.theta[1], self.theta[2], self.theta[3], self.theta[4]
        for t in range(T):
            nsq = n * n
            n = (x[:, t] + theta0 * n + theta1 * nsq + theta2 * nsq * n)
            n = torch.tanh(n) # This nonlinearity helps keep n stable
            outputs[:, t] = theta3 * n + theta4 * nsq
            assert not torch.isnan(n).any(), f"NaN detected at step {t}"
            if return_n:
                n_traj[:, t] = n
        self.last_n_traj = n_traj.detach()

        return outputs

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )
        self.padding = (kernel_size - 1) * dilation
        self.relu = nn.ReLU()
        self.resample = None
        if in_channels != out_channels:
            self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = self.relu(out)
        if self.resample:
            x = self.resample(x)
        return out + x # residual connection

class TCN(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, kernel_size=10, hidden_channels=32):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, kernel_size, dilation)
            )
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        # Calculate the total receptive field for the whole TCN stack
        self.receptive_field = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            self.receptive_field += (kernel_size - 1) * dilation

    def forward(self, xin):
        x = xin.unsqueeze(1)    # [B,1,T]
        out = self.tcn(x)     # [B,H,T]
        out = self.readout(out).squeeze(1)
        out = out - out.mean(dim=1, keepdim=True)  # [B,T]
        return out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params

class TCN_channel(nn.Module):
    def __init__(self, nlayers=3, dilation_base=2, kernel_size=10,
                 hidden_channels=32, learn_noise=False, gaussian=True):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            layers.append(
                TCNBlock(in_channels, hidden_channels, kernel_size, dilation)
            )
            in_channels = hidden_channels
        self.learn_noise = learn_noise
        self.tcn = nn.Sequential(*layers)
        if gaussian:
            self.readout = nn.Conv1d(hidden_channels, 2, kernel_size=1) # 2 channels mean | std
        else:
            self.readout = nn.Conv1d(hidden_channels, 3, kernel_size=1) # 3 channels mean | std | nu
        self.kernel_size = kernel_size
        self.gaussian = gaussian

        # Calculate the total receptive field for the whole TCN stack
        self.receptive_field = 1
        for i in range(nlayers):
            dilation = dilation_base ** i
            self.receptive_field += (kernel_size - 1) * dilation

        if not gaussian:
            with torch.no_grad():
                # Initialize nu bias towards Gaussian for stability
                self.readout.bias[2].fill_(48)


    def sample_student_t_pytorch(self, mean, std, nu):
        """
        Samples from a Student's t-distribution using PyTorch's built-in implementation.
        Uses rsample() to maintain gradients for 'std' and 'nu'.
        """
        nu = torch.clamp(nu, min=2.001)
        std = torch.clamp(std, min=1e-6)
        dist = StudentT(df=nu, loc=mean, scale=std)
        return dist.rsample()


    def sample_student_t_mps(self, mean, std, nu):
        '''
        Wilson-Hilferty Approximation for chi^2 converted to scaled and shifted student t
        '''
        z = torch.randn_like(mean)
        z_chi = torch.randn_like(mean)
        chi2_approx = nu * (1 - 2/(9*nu) + z_chi * torch.sqrt(2/(9*nu))).pow(3)
        scale = torch.sqrt(nu / (chi2_approx + 1e-6))
        return mean + std * z * scale


    def forward(self, xin):
        x = xin.unsqueeze(1)    # [B,1,T]
        out = self.tcn(x)     # [B,H,T]
        out = self.readout(out) # [B, 3, T] mean | std | nu
        mean_out = out[:, 0, :]
        log_std_out = out[:, 1, :]
        std_out = torch.exp(log_std_out)
        if not self.gaussian:
            log_nu_out = out[:, 2, :]
            nu_out = torch.nn.functional.softplus(log_nu_out)
            nu_out = torch.clamp(nu_out, 2, 50) # nu between 2 and 50
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)  # [B ,T]

        # # Produce noisy output
        if self.gaussian:
            z = torch.randn_like(mean_out)
            noisy_out = mean_out + std_out * z
            nu_out = torch.full_like(mean_out, float('inf')) # nu = inf for Gaussian
        else:
            if xin.device.type == "mps":
                noisy_out = self.sample_student_t_mps(mean_out, std_out, nu_out)
            else:
                noisy_out = self.sample_student_t_pytorch(mean_out, std_out, nu_out)

        if self.learn_noise:
            return noisy_out, mean_out, std_out, nu_out
        else:
            return mean_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


def _complex_diag_scan(a_re, a_im, b_re, b_im):
    '''Inclusive parallel prefix scan of the diagonal linear recurrence
        x_k = a_k * x_{k-1} + b_k          (all complex, elementwise / diagonal)
    for every batch and state channel at once, using the Hillis-Steele doubling
    scheme (O(T log T), fully differentiable, no Python loop over time).

    a_*, b_* have shape [*, T, N] (a may broadcast on the batch dim). The monoid
    combine is the same associative operator as the reference LRU parallel scan:
    composing (a_i, b_i) then (a_j, b_j) gives (a_j a_i, a_j b_i + b_j). Returns the
    scanned additive component (b), i.e. the state sequence x_k, as (re, im).

    Complex numbers are carried as separate real/imag tensors so the whole thing
    runs on MPS, which lacks solid native complex support (cf. the Student-t path).
    '''
    T = a_re.shape[-2]
    d = 1
    while d < T:
        # shift right by d along time; the missing left neighbour is the monoid
        # identity (multiplier 1, addend 0), so pad a with 1 and b with 0.
        pa_re = F.pad(a_re, (0, 0, d, 0), value=1.0)[..., :T, :]
        pa_im = F.pad(a_im, (0, 0, d, 0), value=0.0)[..., :T, :]
        pb_re = F.pad(b_re, (0, 0, d, 0), value=0.0)[..., :T, :]
        pb_im = F.pad(b_im, (0, 0, d, 0), value=0.0)[..., :T, :]

        # new_a = a * pa ; new_b = a * pb + b   (complex mult, current=right operand)
        new_a_re = a_re * pa_re - a_im * pa_im
        new_a_im = a_re * pa_im + a_im * pa_re
        new_b_re = a_re * pb_re - a_im * pb_im + b_re
        new_b_im = a_re * pb_im + a_im * pb_re + b_im

        a_re, a_im, b_re, b_im = new_a_re, new_a_im, new_b_re, new_b_im
        d *= 2
    return b_re, b_im


class LRU(nn.Module):
    '''Linear Recurrent Unit layer (Orvieto et al., 2023). A linear diagonal
    complex-state recurrence run with a parallel scan, plus a skip-connected
    output projection.

    The complex arithmetic is kept as explicit real/imag pairs for MPS support.

    N = state_dim (recurrent state size), H = model_dim (in/out feature size).
    Input/output are real sequences of shape [B, T, H].
    '''
    def __init__(self, state_dim, model_dim, r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        N, H = state_dim, model_dim
        self.N = N
        self.H = H

        # Lambda ~ uniform on the complex ring between r_min and r_max, phase in
        # [0, max_phase]; stored as its stable log-parametrisation (nu, theta).
        u1 = torch.rand(N)
        u2 = torch.rand(N)

        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(max_phase * u2)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)

        # Glorot-initialised input/output projections (real + imag parts).
        self.B_re = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.B_im = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.C_re = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.C_im = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.randn(H))

        # Normalization of the recurrence
        diag_lambda = torch.exp(-torch.exp(nu_log) + 1j * torch.exp(theta_log))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2) + 1e-8)
        self.gamma_log = nn.Parameter(gamma_log)

    def forward(self, u):
        # u: [B, T, H]
        B, T, _ = u.shape
        N = self.N

        # Materialize diagonal lambda 
        modulus = torch.exp(-torch.exp(self.nu_log))          # [N]
        phase = torch.exp(self.theta_log)                     # [N]
        lam_re = modulus * torch.cos(phase)
        lam_im = modulus * torch.sin(phase)

        # Normalized input projection B_norm = (B_re + iB_im) * exp(gamma).
        gamma = torch.exp(self.gamma_log).unsqueeze(-1)       # [N, 1]
        Bn_re = self.B_re * gamma
        Bn_im = self.B_im * gamma

        # Bu_k = B_norm @ u_k for the whole sequence (u is real): [B, T, N].
        Bu_re = u @ Bn_re.t()
        Bu_im = u @ Bn_im.t()

        # Lambda is time-invariant: broadcast it across time (batch dim stays 1).
        a_re = lam_re.view(1, 1, N).expand(1, T, N)
        a_im = lam_im.view(1, 1, N).expand(1, T, N)
        x_re, x_im = _complex_diag_scan(a_re, a_im, Bu_re, Bu_im)  # states x_k

        # y_k = Re(C @ x_k) + D * u_k.
        y = x_re @ self.C_re.t() - x_im @ self.C_im.t()       # [B, T, H]
        y = y + u * self.D
        return y


class LRUBlock(nn.Module):
    '''One residual LRU block: pre-norm -> LRU -> GELU -> GLU, as in the paper.'''
    def __init__(self, state_dim, model_dim, dropout=0.0,
                 r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.lru = LRU(state_dim, model_dim, r_min, r_max, max_phase)
        self.glu_w = nn.Linear(model_dim, model_dim)
        self.glu_v = nn.Linear(model_dim, model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = self.lru(z)
        z = self.drop(F.gelu(z))
        z = self.glu_w(z) * torch.sigmoid(self.glu_v(z))      # gated linear unit
        z = self.drop(z)
        return x + z                                          # residual connection


class LRU_channel(nn.Module):
    '''Stacked Linear Recurrent Unit channel model, a modern deep state-space /
    linear-RNN baseline to sit alongside TCN_channel and the memory polynomials.

    Mirrors TCN_channel's I/O contract so it drops into the same grid-search
    adapter: input [B, T], and either returns a centred deterministic estimate
    `mean_out` (learn_noise=False) or the probabilistic tuple
    (noisy_out, mean_out, std_out, nu_out) with a Gaussian (gaussian=True) or
    Student-t (gaussian=False) observation model (learn_noise=True).
    '''
    def __init__(self, state_dim=64, hidden_dim=64, n_layers=2, dropout=0.0,
                 r_min=0.0, r_max=1.0, max_phase=6.28,
                 learn_noise=False, gaussian=True):
        super().__init__()
        self.learn_noise = learn_noise
        self.gaussian = gaussian

        self.encoder = nn.Linear(1, hidden_dim)               # per-timestep 1 -> H
        self.blocks = nn.ModuleList([
            LRUBlock(state_dim, hidden_dim, dropout, r_min, r_max, max_phase)
            for _ in range(n_layers)
        ])
        # readout channels: mean | (mean, std) | (mean, std, nu)
        if not learn_noise:
            out_channels = 1
        elif gaussian:
            out_channels = 2
        else:
            out_channels = 3
        self.readout = nn.Linear(hidden_dim, out_channels)
        self.receptive_field = 1

        if learn_noise and not gaussian:
            with torch.no_grad():
                # Bias nu large so the Student-t starts near-Gaussian (stability).
                self.readout.bias[2].fill_(48)

    def sample_student_t_pytorch(self, mean, std, nu):
        '''Samples from a Student's t via PyTorch's built-in dist; rsample()
        keeps gradients flowing through std and nu.'''
        nu = torch.clamp(nu, min=2.001)
        std = torch.clamp(std, min=1e-6)
        dist = StudentT(df=nu, loc=mean, scale=std)
        return dist.rsample()

    def sample_student_t_mps(self, mean, std, nu):
        '''Wilson-Hilferty chi^2 approximation for a scaled/shifted Student-t
        (StudentT.rsample is unsupported on MPS).'''
        z = torch.randn_like(mean)
        z_chi = torch.randn_like(mean)
        chi2_approx = nu * (1 - 2 / (9 * nu) + z_chi * torch.sqrt(2 / (9 * nu))).pow(3)
        scale = torch.sqrt(nu / (chi2_approx + 1e-6))
        return mean + std * z * scale

    def forward(self, xin):
        # xin: [B, T]
        x = self.encoder(xin.unsqueeze(-1))                   # [B, T, H]
        for block in self.blocks:
            x = block(x)
        out = self.readout(x)                                 # [B, T, out_channels]

        mean_out = out[..., 0]                                # [B, T]
        mean_out = mean_out - mean_out.mean(dim=1, keepdim=True)
        if not self.learn_noise:
            return mean_out

        std_out = torch.exp(out[..., 1])
        if self.gaussian:
            z = torch.randn_like(mean_out)
            noisy_out = mean_out + std_out * z
            nu_out = torch.full_like(mean_out, float('inf'))  # nu = inf for Gaussian
        else:
            nu_out = torch.nn.functional.softplus(out[..., 2])
            nu_out = torch.clamp(nu_out, 2, 50)               # nu between 2 and 50
            if xin.device.type == "mps":
                noisy_out = self.sample_student_t_mps(mean_out, std_out, nu_out)
            else:
                noisy_out = self.sample_student_t_pytorch(mean_out, std_out, nu_out)

        return noisy_out, mean_out, std_out, nu_out

    def get_num_params(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params


class memory_polynomial_channel(nn.Module):
    def __init__(self,
                 weights,
                memory_linear,
                memory_nonlinear,
                nonlinearity_order,
                device
                 ):
        super().__init__()
        if device == torch.device('mps'):
            print("MPS not supported. . . switching to CPU")
            device = torch.device('cpu')
        if weights:
            self.weights = torch.tensor(weights, device=device)
        else:
            self.weights = None
        self.memory_linear = memory_linear
        self.memory_nonlinear = memory_nonlinear
        self.nonlinearity_order = nonlinearity_order
        self.device = device

    def get_num_regressors(self):
        return (self.memory_linear + 1) + (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1)

    def _iter_feature_chunks(self, X, Y, batch_chunk):
        # Chunk over the BATCH dim so each chunk holds complete sequences
        B, T = X.shape
        for b0 in range(0, B, batch_chunk):
            Xb = X[b0:b0 + batch_chunk]                 # [bc, T]
            A_blk = self._create_regressors(Xb)         # [bc*T, P] only this chunk lives in memory
            y_blk = Y[b0:b0 + batch_chunk].reshape(-1)  # [bc*T]
            yield A_blk, y_blk

    def _fit_normal_eqs(self, X, Y, batch_chunk=8, ridge=0.0):
        P = self.get_num_regressors()
        G = torch.zeros(P, P, dtype=torch.float64, device=self.device)
        c = torch.zeros(P,    dtype=torch.float64, device=self.device)
        for A_blk, y_blk in self._iter_feature_chunks(X, Y, batch_chunk):
            A_blk = A_blk.double()
            y_blk = y_blk.double()
            G += A_blk.T @ A_blk
            c += A_blk.T @ y_blk
            del A_blk, y_blk            # release the chunk before building the next one
        G += ridge * torch.eye(P, dtype=torch.float64, device=self.device)
        self.weights = torch.linalg.solve(G, c)

    def _create_regressors(self, X):
        B, T = X.shape
        # Each example and target will get a matrix and column vector. All will be stacked
        # to form a A with shape [NxT, memory_linear + memory_nonlinearxnonlinear_order] regressor matrix
        batched_regressor_cols = []
        num_regressors = (
            (self.memory_linear + 1) +
            (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1)
        )
        regressor_length = T * B
        for i in range(self.memory_linear + 1):
            X_shifted = torch.roll(X, i, dims=1)
            X_shifted[:, :i] = 0.0
            batched_regressor_cols.append(X_shifted)

        for k in range(2, self.nonlinearity_order + 1):
            for j in range(self.memory_nonlinear + 1):
                X_shifted = torch.roll(X, j, dims=1)
                X_shifted[:, :j] = 0.0
                batched_regressor_cols.append(torch.pow(X_shifted, k))

        stack = torch.stack(batched_regressor_cols) # [features, B, T]
        stack = stack.permute(1, 2, 0) # [B, T, freatures]
        A = stack.reshape(regressor_length, num_regressors)
        return A.to(self.device)

    def show_terms(self, plot=False):
        weights = self.weights.detach().cpu()
        terms = []
        linear_weights = []
        idx = 0
        for i in range(self.memory_linear + 1):
            terms.append(f"x[{-i}]")
            linear_weights.append(weights[idx].item())
            idx += 1
        if plot:
            plt.plot(linear_weights)
            plt.title("Plot of Linear Weights vs. Memory Length")
            plt.xlabel("Memory Tap")
            plt.ylabel("Weight Value")
            plt.show()

        for k in range(2, self.nonlinearity_order + 1):
            k_th_weights = []
            for j in range(self.memory_nonlinear + 1):
                terms.append(f"x[{-j}]^{k}")
                k_th_weights.append(weights[idx].item())
                idx += 1

            if plot:
                plt.plot(k_th_weights)
                plt.title(f"Plot of Weights Order {k} vs. Memory Length")
                plt.xlabel("Memory Tap")
                plt.ylabel("Weight Value")
                plt.show()

        weights = None
        if self.weights is not None:
            weights = self.weights.detach().cpu().tolist()

        return terms, weights

    @torch.no_grad()
    def calculate_err(self, X, Y, batch_chunk=8, plot=False):
        X = X.to(self.device)
        Y = Y.to(self.device)
        n_regressors = self.get_num_regressors()
        G = torch.zeros(n_regressors, n_regressors, dtype=torch.float64, device=self.device)
        c = torch.zeros(n_regressors, dtype=torch.float64, device=self.device)
        total_variance = torch.zeros((), dtype=torch.float64, device=self.device)
        for A_blk, y_blk in self._iter_feature_chunks(X, Y, batch_chunk):
            A_blk = A_blk.double()
            y_blk = y_blk.double()
            G += A_blk.T @ A_blk
            c += A_blk.T @ y_blk
            total_variance += (y_blk * y_blk).sum()
            del A_blk, y_blk
        # Gram = R^T R via Cholesky decomposition; then g = Q^T b = R^{-T} (A^T b) = R^{-T} c => R^{T} g = c
        # which is a simple triangular system to solve for g, which gives us the variance contribution of each regressor. Then we can calculate ERR as (g_i^2 / total_variance) * 100
        R = torch.linalg.cholesky(G, upper=True)
        g = torch.linalg.solve_triangular(R.T, c.unsqueeze(-1), upper=False).squeeze(-1)
        component_variances = g ** 2
        terms, _ = self.show_terms(plot=False)
        ERR_values = (component_variances / total_variance) * 100
        err_list = ERR_values.cpu().tolist()

        num_linear = self.memory_linear + 1
        total_linear_err = torch.sum(ERR_values[:num_linear]).item()
        total_nonlinear_err = torch.sum(ERR_values[num_linear:]).item()
        ranked_data = list(zip(terms, err_list))
        ranked_data.sort(key=lambda x: x[1], reverse=True) # sort by ERR magnitude
        if plot:
            print("-" * 50)
            print(f"{'Rank':<5} | {'Term String':<20} | {'ERR (%)':<15}")
            print("-" * 50)

            cumulative_err = 0.0
            for i, (term, err) in enumerate(ranked_data):
                cumulative_err += err
                print(f"{i+1:<5} | {term:<20} | {err:.6f}%")

            print("-" * 50)
            print(f"Total Variance Explained: {cumulative_err:.4f}%")
            print(f"  > Linear Contribution:    {total_linear_err:.4f}%")
            print(f"  > Nonlinear Contribution: {total_nonlinear_err:.4f}%")
            print("-" * 50)
        return terms, ERR_values


    def fit(self, X, Y, batch_chunk=8, ridge=0.0):
        X = X.to(self.device); Y = Y.to(self.device)
        self._fit_normal_eqs(X, Y, batch_chunk=batch_chunk, ridge=ridge)
        self.weights = self.weights.to(X.dtype)
        return self.weights

    @torch.no_grad()
    def predict(self, X, batch_chunk=8):     # chunked forward
        X = X.to(self.device)
        B, T = X.shape
        out = torch.empty(B, T, dtype=self.weights.dtype, device=self.device)
        for b0 in range(0, B, batch_chunk):
            A_blk = self._create_regressors(X[b0:b0 + batch_chunk])
            out[b0:b0 + batch_chunk] = (A_blk @ self.weights).reshape(-1, T)
            del A_blk
        return out

    def forward(self, X):
        A_x = self._create_regressors(X)
        B, T = X.shape
        y_pred = A_x @ self.weights
        y_pred = y_pred.reshape(B, T)
        return y_pred

    def get_num_params(self):
        assert self.weights is not None, "Model must be fitted before getting number of parameters."
        return self.weights.numel()
    

class GeneralizedMemoryPolynomial(memory_polynomial_channel):
    def __init__(self, weights, memory_linear, memory_nonlinear, nonlinearity_order, cross_term_depth, device):
        super().__init__(weights, memory_linear, memory_nonlinear, nonlinearity_order, device)
        self.cross_term_depth = cross_term_depth

    def get_num_regressors(self):
        return (
            (self.memory_linear + 1)
            + (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1) * (self.cross_term_depth + 1)
        )

    def _create_regressors(self, X):
        B, T = X.shape
        # Each example and target will get a matrix and column vector. All will be stacked
        # to form a A with shape [NxT, memory_linear + memory_nonlinearxnonlinear_order] regressor matrix

        X_powers = {k: torch.pow(X, k) for k in range(1, self.nonlinearity_order + 1)}
        batched_regressor_cols = []
        num_regressors = (
            (self.memory_linear + 1) +
            (self.memory_nonlinear + 1) * (self.nonlinearity_order - 1) * (self.cross_term_depth + 1)
        )
        regressor_length = T * B
        for i in range(self.memory_linear + 1):
            X_shifted = torch.roll(X, i, dims=1)
            X_shifted[:, :i] = 0.0
            batched_regressor_cols.append(X_shifted)

        for k in range(2, self.nonlinearity_order + 1):
            X_pow_k = X_powers[k]
            for j in range(self.memory_nonlinear + 1):
                X_shifted_pow = torch.roll(X_pow_k, j, dims=1)
                X_shifted_pow[:, :j] = 0.0
                batched_regressor_cols.append(X_shifted_pow)
                
        # Cross-terms
        for k in range(2, self.nonlinearity_order + 1):
            X_pow_k = X_powers[k - 1]
            for j in range(self.memory_nonlinear + 1):
                X_shifted_base = torch.roll(X, j, dims=1)
                X_shifted_base[:, : (j)] = 0.0
                for d in range(1, self.cross_term_depth + 1):
                    X_shifted_pow = torch.roll(X_pow_k, j + d, dims=1)
                    X_shifted_pow[:, : (j + d)] = 0.0
                    cross_term = X_shifted_pow * X_shifted_base
                    batched_regressor_cols.append(cross_term)

        stack = torch.stack(batched_regressor_cols) # [features, B, T]
        stack = stack.permute(1, 2, 0) # [B, T, features]
        A = stack.reshape(regressor_length, num_regressors)
        return A.to(self.device)

    def show_terms(self, plot=False):
        weights = self.weights.detach().cpu()
        terms = []
        linear_weights = []
        idx = 0
        for i in range(self.memory_linear + 1):
            terms.append(f"x[{-i}]")
            linear_weights.append(weights[idx].item())
            idx += 1
        if plot:
            plt.plot(linear_weights)
            plt.title("Plot of Linear Weights vs. Memory Length")
            plt.xlabel("Memory Tap")
            plt.ylabel("Weight Value")
            plt.show()

        for k in range(2, self.nonlinearity_order + 1):
            k_th_weights = []
            for j in range(self.memory_nonlinear + 1):
                terms.append(f"x[{-j}]^{k}")
                k_th_weights.append(weights[idx].item())
                idx += 1

            if plot:
                plt.plot(k_th_weights)
                plt.title(f"Plot of Weights Order {k} vs. Memory Length")
                plt.xlabel("Memory Tap")
                plt.ylabel("Weight Value")
                plt.show()

        # Cross-terms
        for k in range(2, self.nonlinearity_order + 1):
            for j in range(self.memory_nonlinear + 1):
                for d in range(1, self.cross_term_depth + 1):
                    terms.append(f"x[{-(j)}] * x[{-(j + d)}]^{k - 1}")
                    idx += 1

        weights = None
        if self.weights is not None:
            weights = self.weights.detach().cpu().tolist()

        return terms, weights