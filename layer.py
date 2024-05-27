import math
import torch
import statistics
import torch.nn as nn
from torch import empty
from torch.nn import Parameter

def ff_layer(in_features: int, out_features: int, activation_function: torch.nn.functional.relu, learning_rate: float, threshold: float, epochs: int, device: str="cuda"):
    layer_parameter = []
    weight = Parameter(empty((out_features, in_features), device=device))
    bias = Parameter(empty(out_features, device=device))

    def weight_and_bias_initialization():
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
    weight_and_bias_initialization()

    def linear_computation(x: torch.Tensor):
        return torch.matmul(x, weight.t()) + bias

    def forward_pass(x: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return activation_function(linear_computation(x))
    
    layer_parameter.extend([weight, bias])
    optimizer = torch.optim.Adam(layer_parameter, learning_rate)
    def calculate_layer_loss(positive, negative):
        positive_output_features = forward_pass(positive)
        negative_output_features = forward_pass(negative)
        squared_average_activation_positive_phase = positive_output_features.pow(2).mean(1)
        squared_average_activation_negative_phase = negative_output_features.pow(2).mean(1)
        layer_loss = torch.log(1 + torch.exp(torch.cat([
                    -squared_average_activation_positive_phase + threshold,
                    squared_average_activation_negative_phase - threshold]))).mean()
        #TODO: Inverse (squared_average_activation_positive_phase.mean() - squared_average_activation_positive_phase.mean())
        # to turn large value tensor close to zero and turn small value tensor far from zero.
        optimizer.zero_grad()
        layer_loss.backward()
        optimizer.step()
        return layer_loss

    def propagate_positive_and_negative_phase_in_layer(positive_data, negative_data):
        while True:
            compare_number = 0
            loss = 1.0
            while loss > compare_number:
                layer_loss = calculate_layer_loss(positive_data, negative_data)
                loss = round(layer_loss.item(), 3)
                if loss < compare_number:
                    for _ in range(5):
                        layer_loss = calculate_layer_loss(positive_data, negative_data)
                        if round(layer_loss.item(), 3) > compare_number: break
                else:
                    continue
            break

        # detach output layer from gradient flow to next layer
        return forward_pass(positive_data).detach(), forward_pass(negative_data).detach()

    return propagate_positive_and_negative_phase_in_layer, forward_pass
