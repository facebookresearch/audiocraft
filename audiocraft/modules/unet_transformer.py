import torch
import typing as tp
from .transformer import StreamingTransformer, create_sin_embedding


class UnetTransformer(StreamingTransformer):
    """U-net Transformer for processing sequences with optional skip connections.
    This transformer architecture incorporates U-net style skip connections
    between layers, which can be optionally enabled. It inherits from a
    StreamingTransformer.

    Args:
        d_model (int): Dimension of the model, typically the number of expected features in the input.
        num_layers (int): Total number of layers in the transformer.
        skip_connections (bool, optional): Flag to determine whether skip connections should be used.
                                           Defaults to False.
        layer_dropout_p (float, Optional): if given, defined bernoulli prob. to drop a skip connection (in training).
        **kwargs: Additional keyword arguments inherited from `nn.StreamingTransformer`.
    """
    def __init__(self, d_model: int, num_layers: int, skip_connections: bool = False,
                 layer_dropout_p: tp.Optional[float] = None, **kwargs):
        super().__init__(d_model=d_model,
                         num_layers=num_layers,
                         **kwargs)
        self.skip_connect = skip_connections
        if self.skip_connect:
            self.skip_projections = torch.nn.ModuleList([torch.nn.Linear(d_model * 2, d_model)
                                                        for _ in range(num_layers // 2)])
        self.num_layers = num_layers
        self.layer_drop_p = max(min(layer_dropout_p, 1.), 0.) if layer_dropout_p is not None else 0.0

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        if 'offsets' in self._streaming_state:
            offsets = self._streaming_state['offsets']
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)

        if self.positional_embedding in ['sin', 'sin_rope']:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        skip_connections: tp.List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            if self.skip_connect and i >= self.num_layers // 2:

                # in the second half of the layers, add residual connection
                # and linearly project the concatenated features back to d_model
                x = torch.cat([x, skip_connections.pop()], dim=-1)
                x = self.skip_projections[i % len(self.skip_projections)](x)

            x = self._apply_layer(layer, x, *args, **kwargs)

            if self.skip_connect and i < self.num_layers // 2:
                if self.training and torch.rand(1,) < self.layer_drop_p:  # drop skip
                    skip_connections.append(torch.zeros_like(x))
                else:
                    skip_connections.append(x)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return x
