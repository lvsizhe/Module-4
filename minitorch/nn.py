from .fast_ops import FastOps
from .tensor_functions import zeros, rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw).permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return (output, new_height, new_width)


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4).view(batch, channel, new_height, new_width)

    return pooled 


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        out = max_reduce(input, dim)
        out_argmax = (out == input)
        ctx.save_for_backward(out_argmax)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        out_argmax = ctx.saved_values
        return out_argmax * grad_output


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    e = input.exp()
    t = e.sum(dim=dim)
    return e / t


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    m = max(input, dim)
    e = (input - m).exp()
    s = e.sum(dim=dim)
    return input - s.log() - m


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, 4).view(batch, channel, new_height, new_width)
    return pooled


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    if ignore:
        return input
    ratios = rand(input.shape)
    return input * (ratios > rate)
