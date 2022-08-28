class ReLU(Module):
  __parameters__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    argument_1: Tensor) -> Tensor:
    return torch.relu(argument_1)
class Softmax(Module):
  __parameters__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.Softmax,
    argument_1: Tensor) -> Tensor:
    return torch.softmax(argument_1, 1, None)
