import os, sys, time
sys.path.append(os.path.split(__file__)[0])
import torch
import numpy as np
import torchac
from factorized_entropy_model import RoundNoGradient, UniverseQuantR, quantize_ste, Low_bound


class SymmetricConditional(torch.nn.Module):
    def __init__(self, distribution='laplace'):
        super(SymmetricConditional, self).__init__()
        self._likelihood_bound = 1e-9
        self._range_coder_precision = 16
        self.distribution = distribution
    
    def _quantize(self, inputs, mode="noise", step=1.0):
        """Add noise or quantize.
            only noise support differentiable.
        """
        inputs = inputs/step
        if mode is None or mode=="None":
            inputs = inputs
        elif mode=="noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            inputs = inputs + noise
        elif mode=="straight":
            inputs = quantize_ste(inputs)
        elif mode=="UniverseQuant":
            inputs = UniverseQuantR.apply(inputs)
        elif mode=="symbols":
            inputs = RoundNoGradient.apply(inputs)
        inputs = inputs*step

        return inputs

    def _likelihood(self, inputs, loc, scale, quantize_step=1.0):
        """ Estimate the likelihoods conditioned on assumed distribution.
        Arguments: inputs;(quantized); loc; scale;
        Return: likelihood.
        """
        if not (isinstance(quantize_step, float) or isinstance(quantize_step, int)):
            assert quantize_step.shape==inputs.shape
        if self.distribution=='normal':
            m = torch.distributions.normal.Normal(loc, scale)
        if self.distribution=='laplace':
            m = torch.distributions.laplace.Laplace(loc, scale)
        lower = m.cdf(inputs - 0.5 * quantize_step)
        upper = m.cdf(inputs + 0.5 * quantize_step)
        likelihood = torch.abs(upper - lower)
        
        return likelihood

    def forward(self, inputs, loc, scale, quantize_mode="straight", quantize_step=1.0):
        """ Argument: inputs; mean, scale; training;      
        Returns: output quantized tensor; likelihoods.
        """
        if not (isinstance(quantize_step, float) or isinstance(quantize_step, int)):
            assert quantize_step.shape==inputs.shape
        outputs = self._quantize(inputs, mode=quantize_mode, step=quantize_step)
        likelihood = self._likelihood(outputs, loc, scale, quantize_step=1.0)
        likelihood = Low_bound.apply(likelihood)
        
        return outputs, likelihood
        
    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    def _get_cdf(self, loc, scale, min_v, max_v):
        """Get quantized cdf for compress/decompress.
        """
        scale = scale.to(loc.device)
        min_v = min_v.to(loc.device); max_v = max_v.to(loc.device)
        symbols = torch.arange(min_v.int(), max_v.int()+1).reshape(1,1,-1).to(loc.device)
        symbols = symbols.repeat(loc.shape[0], self._channels, 1)# [N, C, S]
        pmf = self._likelihood(symbols.float(), loc, scale)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        cdf = self._pmf_to_cdf(pmf)

        return cdf

    def compress(self, inputs, loc, scale):
        # quantize
        values = self._quantize(inputs, mode="symbols").detach()
        self._channels = values.shape[-1]
        # get symbols
        min_v = values.min().detach().float()
        max_v = values.max().detach().float()
        loc, scale = loc.unsqueeze(2), scale.unsqueeze(2) # [N, C, 1]
        cdf = self._get_cdf(loc, scale, min_v, max_v)
        values -= min_v
        cdf = cdf.cpu()
        values = values.cpu()
        strings = torchac.encode_float_cdf(cdf, values.to(torch.int16), check_input_bounds=True)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy() 

    def decompress(self, strings, loc, scale, min_v, max_v, shape=None, channels=None):
        """Decompress values from their compressed string representations.
        Arguments: strings: A string `Tensor` vector containing the compressed data.(numpy)
        min_v & max_v.(numpy) loc, scale, min_v, max_v.
        shape: A `Tensor` vector of int32 type. Contains the shape of the tensor to be
            decompressed, excluding the batch dimension. [points, channels] (numpy)
        Returns: The decompressed `Tensor`. (torch)  # TODO
        """   
        min_v, max_v = torch.tensor(min_v), torch.tensor(max_v)
        loc, scale = loc.unsqueeze(2), scale.unsqueeze(2) # [N, C, 1]
        cdf = self._get_cdf(loc, scale, min_v, max_v)
        cdf = cdf.cpu()
        # arithmetic decoding
        values = torchac.decode_float_cdf(cdf, strings).float()
        values += min_v.float()

        return values
