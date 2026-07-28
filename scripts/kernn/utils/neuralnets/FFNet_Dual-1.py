import torch
import torch.nn as nn


class FFNet(nn.Module):
    '''
    Dual-input Feed-Forward Neural Network:
    - One branch for kernel features
    - One branch for dihedral angles
    '''
    def __init__(self, n_input_kernel, n_hidden, n_out):
        super().__init__()

        # Kernel feature branch
        self.kernel_net = nn.Sequential(
            nn.Linear(n_input_kernel, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus()
        )

        # Dihedral angle branch
        self.dihedral_net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Softplus()
        )

        # Final combination
        self.output_net = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, kernel_input, dihedral_input):
        k = self.kernel_net(kernel_input)
        d = self.dihedral_net(dihedral_input)
        x = torch.cat((k, d), dim=1)
        return self.output_net(x)


