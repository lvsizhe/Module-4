from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

BLOCK_DIM = 32
THREADS_PER_BLOCK = BLOCK_DIM * BLOCK_DIM

def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(idx, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)

        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)

        out[out_pos] = fn(in_storage[in_pos])


    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(idx, out_shape, out_index)

        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.

    """
    shmem = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < size:
        shmem[cuda.threadIdx.x] = a[idx]

    if cuda.threadIdx.x > 0:
        return

    cuda.syncthreads()

    t = 0.0
    for i in range(size):
        t += shmem[i]
    out[cuda.blockIdx.x] = t


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        shmem = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
        index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(cuda.blockIdx.x, out_shape, index)
        
        group_idx = index[reduce_dim]
        idx = group_idx * cuda.blockDim.x + cuda.threadIdx.x
        if idx < a_shape[reduce_dim]:
            index[reduce_dim] = idx
            a_pos = index_to_position(index, a_strides)
            shmem[cuda.threadIdx.x] = a_storage[a_pos]

        if cuda.threadIdx.x > 0:
            return

        t = reduce_value
        for i in range(a_shape[reduce_dim]):
            t = fn(t, shmem[i])

        index[reduce_dim] = group_idx
        pos = index_to_position(index, out_strides)
        out[pos] = t

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // THREADS_PER_BLOCK + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    shm_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shm_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if idx_x >= size or idx_y >= size:
        return
    
    pos = index_to_position((idx_x, idx_y), (size, 1))
    shm_a[idx_x][idx_y] = a[pos]
    shm_b[idx_x][idx_y] = b[pos]

    cuda.syncthreads()

    total = 0.0
    for i in range(size):
        total += shm_a[idx_x][i] * shm_b[i][idx_y]
    
    out[pos] = total


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (BLOCK_DIM, BLOCK_DIM)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    idx_z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    shm_c = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shm_c[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

    shm_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shm_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    count = (a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM
    for i in range(count):
        # Block-(?, blockIdx.x, i) in a
        x_a = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x
        y_a = i * BLOCK_DIM + cuda.threadIdx.y
        z_a = (idx_z if out_shape[0] == a_shape[0] else 0)
        if x_a < a_shape[1] and y_a < a_shape[2]:
            pos_a = index_to_position((z_a, x_a, y_a), a_strides)
            shm_a[cuda.threadIdx.x][cuda.threadIdx.y] = a_storage[pos_a]
        else:
            shm_a[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        # Block (?, i, blockIdx.y) in b
        x_b = i * BLOCK_DIM + cuda.threadIdx.x
        y_b = cuda.blockIdx.y * BLOCK_DIM + cuda.threadIdx.y
        z_b = (idx_z if out_shape[0] == b_shape[0] else 0)
        if x_b < b_shape[1] and y_b < b_shape[2]:
            pos_b = index_to_position((z_b, x_b, y_b), b_strides)
            shm_b[cuda.threadIdx.x][cuda.threadIdx.y] = b_storage[pos_b]
        else:
            shm_b[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        cuda.syncthreads()
        for j in range(BLOCK_DIM):
            shm_c[cuda.threadIdx.x][cuda.threadIdx.y] += shm_a[cuda.threadIdx.x][j] * shm_b[j][cuda.threadIdx.y]

        cuda.syncthreads()
    
    if idx_z < out_shape[0] and idx_x < out_shape[1] and idx_y < out_shape[2]:
        pos = index_to_position((idx_z, idx_x, idx_y), out_strides)
        out[pos] = shm_c[cuda.threadIdx.x][cuda.threadIdx.y]
        


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (BLOCK_DIM - 1)) // BLOCK_DIM,
        (out.shape[2] + (BLOCK_DIM - 1)) // BLOCK_DIM,
        out.shape[0],
    )
    threadsperblock = (BLOCK_DIM, BLOCK_DIM, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
