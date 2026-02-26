import torch
import torch.nn as nn

NEURON_COUNT = 128

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, NEURON_COUNT)
        self.affine2 = nn.Linear(NEURON_COUNT, NEURON_COUNT)

        self.action_mean = nn.Linear(NEURON_COUNT, num_outputs)

        # Values from pendulum example
        #self.action_mean.weight.data.mul_(0.1)
        #self.action_mean.bias.data.mul_(0.0)

        # Initialize weights using the initialization for ReLU activations in stable-baselines3
        module_gains = {
            self.affine1: torch.math.sqrt(2),
            self.affine2: torch.math.sqrt(2),
            self.action_mean: 0.01  
        }
        for layer, gain in module_gains.items():
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        #self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.action_log_std = nn.Parameter(torch.full((1, num_outputs), -0.5))

        #self.saved_actions = []
        #self.rewards = []
        #self.final_value = 0

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))

        action_mean = torch.tanh(self.action_mean(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        # clamp to avoid division by zero when calculating log_prob
        action_log_std = torch.clamp(action_log_std, min=-20.0, max=2.0)
        action_std = torch.exp(action_log_std) 

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, NEURON_COUNT)
        self.affine2 = nn.Linear(NEURON_COUNT, NEURON_COUNT)
        self.value_head = nn.Linear(NEURON_COUNT, 1)
        
        # Values from pendulum example
        #self.value_head.weight.data.mul_(0.1)
        #self.value_head.bias.data.mul_(0.0)

        # Initialize weights using the initialization for ReLU activations in stable-baselines3
        module_gains = {
            self.affine1: torch.math.sqrt(2),
            self.affine2: torch.math.sqrt(2),
            self.value_head: 1.0
        }

        for layer, gain in module_gains.items():
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
    
class CostValue(nn.Module):
    def __init__(self, num_inputs):
        super(CostValue, self).__init__()
        self.affine1 = nn.Linear(num_inputs, NEURON_COUNT)
        self.affine2 = nn.Linear(NEURON_COUNT, NEURON_COUNT)
        self.value_head = nn.Linear(NEURON_COUNT, 1)
        # Values from pendulum example
        #self.value_head.weight.data.mul_(0.1)
        #self.value_head.bias.data.mul_(0.0)

        # Initialize weights using the initialization for ReLU activations in stable-baselines3
        module_gains = {
            self.affine1: torch.math.sqrt(2),
            self.affine2: torch.math.sqrt(2),
            self.value_head: 1.0
        }
        for layer, gain in module_gains.items():
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))

        #state_values = self.softplus(self.value_head(x))
        state_values = self.value_head(x)
        return state_values