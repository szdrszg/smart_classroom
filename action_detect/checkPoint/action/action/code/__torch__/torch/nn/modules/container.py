class Sequential(Module):
  __parameters__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.linear.Linear
  __annotations__["1"] = __torch__.torch.nn.modules.activation.ReLU
  __annotations__["2"] = __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear
  __annotations__["3"] = __torch__.torch.nn.modules.activation.Softmax
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "1")
    _1 = (getattr(self, "0")).forward(input, )
    _2 = getattr(self, "3")
    _3 = (getattr(self, "2")).forward((_0).forward(_1, ), )
    return (_2).forward(_3, )
