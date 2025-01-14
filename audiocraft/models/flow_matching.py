# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp
import torch
from torch import nn
from torchdiffeq import odeint  # type: ignore
from ..modules.streaming import StreamingModule
from ..modules.transformer import create_norm_fn, StreamingTransformerLayer
from ..modules.unet_transformer import UnetTransformer
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningAttributes,
    JascoCondConst
)
from ..modules.jasco_conditioners import JascoConditioningProvider
from ..modules.activations import get_activation_fn

from .lm import ConditionTensors, init_layer


logger = logging.getLogger(__name__)


@dataclass
class FMOutput:
    latents: torch.Tensor  # [B, T, D]
    mask: torch.Tensor  # [B, T]


class CFGTerm:
    """
    Base class for Multi Source Classifier-Free Guidance (CFG) terms. This class represents a term in the CFG process,
    which is used to guide the generation process by adjusting the influence of different conditions.
    Attributes:
        conditions (dict): A dictionary of conditions that influence the generation process.
        weight (float): The weight of the CFG term, determining its influence on the generation.
    """
    def __init__(self, conditions, weight):
        self.conditions = conditions
        self.weight = weight

    def drop_irrelevant_conds(self, conditions):
        """
        Drops irrelevant conditions from the CFG term. This method should be implemented by subclasses.
        Args:
            conditions (dict): The conditions to be filtered.
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("No base implementation for setting generation params.")


class AllCFGTerm(CFGTerm):
    """
    A CFG term that retains all conditions. This class does not drop any condition.
    """
    def __init__(self, conditions, weight):
        super().__init__(conditions, weight)
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        pass


class NullCFGTerm(CFGTerm):
    """
    A CFG term that drops all conditions, effectively nullifying their influence.
    """
    def __init__(self, conditions, weight):
        super().__init__(conditions, weight)
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        """
        Drops all conditions by applying a dropout with probability 1.0, effectively nullifying their influence.
        """
        self.conditions = ClassifierFreeGuidanceDropout(p=1.0)(
                                                        samples=self.conditions,
                                                        cond_types=["wav", "text", "symbolic"])


class TextCFGTerm(CFGTerm):
    """
    A CFG term that selectively drops conditions based on specified dropout probabilities for different types
    of conditions, such as 'symbolic' and 'wav'.
    """
    def __init__(self, conditions, weight, model_att_dropout):
        """
        Initializes a TextCFGTerm with specified conditions, weight, and model attention dropout configuration.
        Args:
            conditions (dict): The conditions to be used in the CFG process.
            weight (float): The weight of the CFG term.
            model_att_dropout (object): The attribute dropouts used by the model.
        """
        super().__init__(conditions, weight)
        if 'symbolic' in model_att_dropout.p:
            self.drop_symbolics = {k: 1.0 for k in model_att_dropout.p['symbolic'].keys()}
        else:
            self.drop_symbolics = {}
        if 'wav' in model_att_dropout.p:
            self.drop_wav = {k: 1.0 for k in model_att_dropout.p['wav'].keys()}
        else:
            self.drop_wav = {}
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        self.conditions = AttributeDropout({'symbolic': self.drop_symbolics,
                                            'wav': self.drop_wav})(self.conditions)  # drop temporal conds


class FlowMatchingModel(StreamingModule):
    """
    A flow matching model inherits from StreamingModule.
    This model uses a transformer architecture to process and fuse conditions, applying learned embeddings and
    transformations and predicts multi-source guided vector fields.
    Attributes:
        condition_provider (JascoConditioningProvider): Provider for conditioning attributes.
        fuser (ConditionFuser): Fuser for combining multiple conditions.
        dim (int): Dimensionality of the model's main features.
        num_heads (int): Number of attention heads in the transformer.
        flow_dim (int): Dimensionality of the flow features.
        chords_dim (int): Dimensionality for chord embeddings, if used.
        drums_dim (int): Dimensionality for drums embeddings, if used.
        melody_dim (int): Dimensionality for melody embeddings, if used.
        hidden_scale (int): Scaling factor for the dimensionality of the feedforward network in the transformer.
        norm (str): Type of normalization to use ('layer_norm' or other supported types).
        norm_first (bool): Whether to apply normalization before other operations in the transformer layers.
        bias_proj (bool): Whether to include bias in the projection layers.
        weight_init (Optional[str]): Method for initializing weights.
        depthwise_init (Optional[str]): Method for initializing depthwise convolutional layers.
        zero_bias_init (bool): Whether to initialize biases to zero.
        cfg_dropout (float): Dropout rate for configuration settings.
        cfg_coef (float): Coefficient for configuration influence.
        attribute_dropout (Dict[str, Dict[str, float]]): Dropout rates for specific attributes.
        time_embedding_dim (int): Dimensionality of time embeddings.
        **kwargs: Additional keyword arguments for the transformer.
    Methods:
        __init__: Initializes the model with the specified attributes and configuration.
    """
    def __init__(self, condition_provider: JascoConditioningProvider,
                 fuser: ConditionFuser,
                 dim: int = 128,
                 num_heads: int = 8,
                 flow_dim: int = 128,
                 chords_dim: int = 0,
                 drums_dim: int = 0,
                 melody_dim: int = 0,
                 hidden_scale: int = 4,
                 norm: str = 'layer_norm',
                 norm_first: bool = False,
                 bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None,
                 depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False,
                 cfg_dropout: float = 0,
                 cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {},
                 time_embedding_dim: int = 128,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef

        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.dim = dim  # transformer dim
        self.flow_dim = flow_dim
        self.chords_dim = chords_dim
        self.emb = nn.Linear(flow_dim + chords_dim + drums_dim + melody_dim, dim, bias=False)
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])

        self.transformer = UnetTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first,
            layer_class=StreamingTransformerLayer,
            **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.linear = nn.Linear(dim, flow_dim, bias=bias_proj)
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

        # init time parameter embedding
        self.d_temb1 = time_embedding_dim
        self.d_temb2 = 4 * time_embedding_dim
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.d_temb1,
                            self.d_temb2),
            torch.nn.Linear(self.d_temb2,
                            self.d_temb2),
        ])
        self.temb_proj = nn.Linear(self.d_temb2, dim)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """
        #######################################################################################################
        TAKEN FROM: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
        #######################################################################################################
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def _embed_time_parameter(self, t: torch.Tensor):
        """
        #######################################################################################################
        TAKEN FROM: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
        #######################################################################################################
        """
        temb = self._get_timestep_embedding(t.flatten(), self.d_temb1)
        temb = self.temb.dense[0](temb)
        temb = temb * torch.sigmoid(temb)  # swish activation
        temb = self.temb.dense[1](temb)
        return temb

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        init_layer(self.emb, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        init_layer(self.linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    def _align_seq_length(self,
                          cond: torch.Tensor,
                          seq_len: int = 500):
        # trim if needed
        cond = cond[:, :seq_len, :]

        # pad if needed
        B, T, C = cond.shape
        if T < seq_len:
            cond = torch.cat((cond, torch.zeros((B, seq_len - T, C), dtype=cond.dtype, device=cond.device)), dim=1)

        return cond

    def forward(self,
                latents: torch.Tensor,
                t: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None) -> torch.Tensor:
        """Apply flow matching forward pass on latents and conditions.
        Given a tensor of noisy latents of shape [B, T, D] with D the flow dim and T the sequence steps,
        and a time parameter tensor t, return the vector field with shape [B, T, D].

        Args:
            latents (torch.Tensor): noisy latents.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: estimated vector field v_theta.
        """
        assert condition_tensors is not None, "FlowMatchingModel require pre-calculation of condition tensors"
        assert not conditions, "Shouldn't pass unprocessed conditions to FlowMatchingModel."

        B, T, D = latents.shape
        x = latents

        # concat temporal conditions on the feature dimension
        temporal_conds = JascoCondConst.ALL.value
        for cond in temporal_conds:
            if cond not in condition_tensors:
                continue
            c = self._align_seq_length(condition_tensors[cond][0], seq_len=T)
            x = torch.concat((x, c), dim=-1)

        # project to transformer dimension
        input_ = self.emb(x)

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        # embed time parameter
        t_embs = self._embed_time_parameter(t)

        # add it to cross_attention_input
        cross_attention_input = cross_attention_input + self.temb_proj(t_embs[:, None, :])

        out = self.transformer(input_, cross_attention_src=cross_attention_input)

        if self.out_norm:
            out = self.out_norm(out)
        v_theta = self.linear(out)  # [B, T, D]

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            v_theta = v_theta[:, :, -T:]

        return v_theta  # [B, T, D]

    def _multi_source_cfg_preprocess(self,
                                     conditions: tp.List[ConditioningAttributes],
                                     cfg_coef_all: float,
                                     cfg_coef_txt: float,
                                     min_weight: float = 1e-6):
        """
        Preprocesses the CFG terms for multi-source conditional generation.
        Args:
            conditions (list): A list of conditions to be applied.
            cfg_coef_all (float): The coefficient for all conditions.
            cfg_coef_txt (float): The coefficient for text conditions.
            min_weight (float): The minimal absolute weight for calculating a CFG term.
        Returns:
            tuple: A tuple containing condition_tensors and cfg_terms.
                condition_tensors is a dictionary or ConditionTensors object with tokenized conditions.
                cfg_terms is a list of CFGTerm objects with weights adjusted based on the coefficients.
        """
        condition_tensors: tp.Optional[ConditionTensors]
        cfg_terms = []
        if conditions:
            # conditional terms
            cfg_terms = [AllCFGTerm(conditions=conditions, weight=cfg_coef_all),
                         TextCFGTerm(conditions=conditions, weight=cfg_coef_txt,
                                     model_att_dropout=self.att_dropout)]

            # add null term
            cfg_terms.append(NullCFGTerm(conditions=conditions, weight=1 - sum([ct.weight for ct in cfg_terms])))

            # remove terms with negligible weight
            for ct in cfg_terms:
                if abs(ct.weight) < min_weight:
                    cfg_terms.remove(ct)

            conds: tp.List[ConditioningAttributes] = sum([ct.conditions for ct in cfg_terms], [])
            tokenized = self.condition_provider.tokenize(conds)
            condition_tensors = self.condition_provider(tokenized)
        else:
            condition_tensors = {}

        return condition_tensors, cfg_terms

    def estimated_vector_field(self, z, t, condition_tensors=None, cfg_terms=[]):
        """
        Estimates the vector field for the given latent variables and time parameter,
        conditioned on the provided conditions.
        Args:
            z (Tensor): The latent variables.
            t (float): The time variable.
            condition_tensors (ConditionTensors, optional): The condition tensors. Defaults to None.
            cfg_terms (list, optional): The list of CFG terms. Defaults to an empty list.
        Returns:
            Tensor: The estimated vector field.
        """
        if len(cfg_terms) > 1:
            z = z.repeat(len(cfg_terms), 1, 1)  # duplicate noisy latents for multi-source CFG
        v_thetas = self(latents=z, t=t, conditions=[], condition_tensors=condition_tensors)
        return self._multi_source_cfg_postprocess(v_thetas, cfg_terms)

    def _multi_source_cfg_postprocess(self, v_thetas, cfg_terms):
        """
        Postprocesses the vector fields generated for each CFG term to combine them into a single vector field.
        Multi source guidance occurs here.
        Args:
            v_thetas (Tensor): The vector fields for each CFG term.
            cfg_terms (list): The CFG terms used.
        Returns:
            Tensor: The combined vector field.
        """
        if len(cfg_terms) <= 1:
            return v_thetas
        v_theta_per_term = v_thetas.chunk(len(cfg_terms))
        return sum([ct.weight * term_vf for ct, term_vf in zip(cfg_terms, v_theta_per_term)])

    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 cfg_coef_all: float = 3.0,
                 cfg_coef_txt: float = 1.0,
                 euler: bool = False,
                 euler_steps: int = 100,
                 ode_rtol: float = 1e-5,
                 ode_atol: float = 1e-5,
                 ) -> torch.Tensor:
        """
        Generate audio latents given a prompt or unconditionally. This method supports both Euler integration
        and adaptive ODE solving to generate sequences based on the specified conditions and configuration coefficients.

        Args:
            prompt (torch.Tensor, optional): Initial prompt to condition the generation. defaults to None
            conditions (List[ConditioningAttributes]): List of conditioning attributes - text, symbolic or audio.
            num_samples (int, optional): Number of samples to generate.
                                         If None, it is inferred from the number of conditions.
            max_gen_len (int): Maximum length of the generated sequence.
            callback (Callable[[int, int], None], optional): Callback function to monitor the generation process.
            cfg_coef_all (float): Coefficient for the fully conditional CFG term.
            cfg_coef_txt (float): Coefficient for text CFG term.
            euler (bool): If True, use Euler integration, otherwise use adaptive ODE solver.
            euler_steps (int): Number of Euler steps to perform if Euler integration is used.
            ode_rtol (float): ODE solver rtol threshold.
            ode_atol (float): ODE solver atol threshold.

        Returns:
            torch.Tensor: Generated latents, shaped as (num_samples, max_gen_len, feature_dim).
        """

        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        condition_tensors, cfg_terms = self._multi_source_cfg_preprocess(conditions, cfg_coef_all, cfg_coef_txt)

        # flow matching inference
        B, T, D = num_samples, max_gen_len, self.flow_dim

        z_0 = torch.randn((B, T, D), device=device)

        if euler:
            # vanilla Euler intergration
            dt = (1 / euler_steps)
            z = z_0
            t = torch.zeros((1, ), device=device)
            for _ in range(euler_steps):
                v_theta = self.estimated_vector_field(z, t,
                                                      condition_tensors=condition_tensors,
                                                      cfg_terms=cfg_terms)
                z = z + dt * v_theta
                t = t + dt
            z_1 = z
        else:
            # solve with dynamic ode integrator (dopri5)
            t = torch.tensor([0, 1.0 - 1e-5], device=device)
            num_evals = 0

            # define ode vector field function
            def inner_ode_func(t, z):
                nonlocal num_evals
                num_evals += 1
                if callback is not None:
                    ESTIMATED_ODE_SOLVER_STEPS = 300
                    callback(num_evals, ESTIMATED_ODE_SOLVER_STEPS)
                return self.estimated_vector_field(z, t,
                                                   condition_tensors=condition_tensors,
                                                   cfg_terms=cfg_terms)

            ode_opts: dict = {"options": {}}
            z = odeint(
                inner_ode_func,
                z_0,
                t,
                **{"atol": ode_atol, "rtol": ode_rtol, **ode_opts},
            )
            logger.info("Generated in %d steps", num_evals)
            z_1 = z[-1]

        return z_1
