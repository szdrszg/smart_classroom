class NetV2(Module):
  __parameters__ = []
  training : bool
  sequential : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.action_detect.net.NetV2,
    input: Tensor) -> Tensor:
    return (self.sequential).forward(input, )
