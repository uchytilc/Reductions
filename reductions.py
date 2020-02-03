import numpy as np
from numba import cuda, float64
import math

#refrence: https://devblogs.nvidia.com/faster-parallel-reductions-kepler/

#Note:
	#for max and min, CUDA's max/min function returns the non-nan value if given only one non-nan value
		#nans are used as defualt value

rthreads = 512

@cuda.jit(device = True)
def __warp_sum_reduce(val):
	mask  = 0xffffffff

	delta = cuda.warpsize//2
	while delta > 0:
		shfl = cuda.shfl_down_sync(mask, val, delta)
		val = val + shfl
		delta //= 2

	return val

@cuda.jit(device = True)
def __block_sum_reduce(val, shared):
	s_n = cuda.threadIdx.x//cuda.warpsize

	val = __warp_sum_reduce(val)

	if cuda.laneid == 0:
		shared[s_n] = val

	cuda.syncthreads()

	val = np.float64(0)
	if cuda.threadIdx.x < (cuda.blockDim.x + cuda.warpsize - 1)//cuda.warpsize:
		val = shared[cuda.laneid]

	if s_n == 0:
		val = __warp_sum_reduce(val)

	return val

@cuda.jit('void(f8[:],f8[:], i8)')
def __sum(_in, _out, size):
	start  = cuda.grid(1)
	stride = cuda.gridsize(1)

	s_shape = 32
	shared  = cuda.shared.array(shape = s_shape, dtype = float64)

	if cuda.threadIdx.x == 0:
		for i in range(s_shape):
			shared[i] = 0

	val = 0
	for n in range(start, size, stride):
		val += _in[n]

	val = __block_sum_reduce(val, shared)

	if cuda.threadIdx.x == 0:
		_out[cuda.blockIdx.x] = val

def sum_gpu(d_in, size  = None, stream = 0):
	if size is None:
		size = d_in.size
	size = min(size, d_in.size)

	threads = min(rthreads, size)
	blocks  = min((size + threads - 1)//threads, 1024)

	#create flattened copy of input
	if len(d_in.shape) > 1:
		d_in = cuda.devicearray.DeviceNDArray(shape = size,
											  strides = (d_in.dtype.itemsize,),
											  dtype = d_in.dtype,
											  gpu_data = d_in.gpu_data,
											  stream = stream)

	d_out = cuda.device_array((blocks,) , dtype = np.float64)

	__sum[blocks, threads, stream](d_in, d_out, size)
	__sum[1, blocks, stream](d_out, d_out, blocks)
	val = d_out[0:1]

	return val.copy_to_host()[0]

@cuda.jit(device = True)
def __warp_min_reduce(val, size):
	mask  = 0xffffffff

	delta = np.int32(cuda.warpsize//2)
	while delta > 0:
		shfl = cuda.shfl_down_sync(mask, val, delta)
		if cuda.grid(1) + delta < size:
			val = min(val, shfl)
		delta //= 2

	return val

@cuda.jit(device = True)
def __block_min_reduce(val, size, shared):
	s_n = cuda.threadIdx.x//cuda.warpsize

	val = __warp_min_reduce(val, size)

	#laneid = position of thread in warp
	if cuda.laneid == 0 and cuda.grid(1) < size:
		shared[s_n] = val

	cuda.syncthreads()

	val = np.nan
	if cuda.threadIdx.x < (cuda.blockDim.x + cuda.warpsize - 1)//cuda.warpsize:
		val = shared[cuda.laneid]

	if s_n == 0:
		val = __warp_min_reduce(val, size)

	return val

@cuda.jit('void(f8[:], f8[:], i8)')
def __min(_in, _out, size):
	start  = cuda.grid(1)
	stride = cuda.gridsize(1)

	s_shape = 32
	shared  = cuda.shared.array(shape = s_shape, dtype = float64)

	if start%cuda.blockDim.x == 0:
		for i in range(s_shape):
			shared[i] = np.nan

	#find mimumum along strided array
	val = _in[start]
	for n in range(start, size, stride):
		val = min(val, _in[n])

	val = __block_min_reduce(val, size, shared)
	if start%cuda.blockDim.x == 0:
		_out[cuda.blockIdx.x] = min(val, _out[cuda.blockIdx.x])

def min_gpu(d_in, size = None, stream = 0):
	if size is None:
		size = d_in.size
	size = min(size, d_in.size)

	threads = min(rthreads, size)
	blocks  = min((size + threads - 1)//threads, 1024)

	#create flattened copy of input
	if len(d_in.shape) > 1:
		d_in = cuda.devicearray.DeviceNDArray(shape = size,
											  strides = (d_in.dtype.itemsize,),
											  dtype = d_in.dtype,
											  gpu_data = d_in.gpu_data)
		
	d_out = cuda.to_device(np.full(blocks, np.nan, dtype = np.float64))

	__min[blocks, threads](d_in, d_out, size)
	__min[1, blocks](d_out, d_out, blocks)
	val = d_out[0:1]

	return val.copy_to_host()[0]

@cuda.jit(device = True)
def __warp_max_reduce(val, size):
	mask  = 0xffffffff

	delta = np.int64(cuda.warpsize//2)
	while delta > 0:
		shfl = cuda.shfl_down_sync(mask, val, delta)
		if cuda.grid(1) + delta < size:
			val = max(val, shfl)
		delta //= 2

	return val

@cuda.jit(device = True)
def __block_max_reduce(val, size, shared):
	s_n = cuda.threadIdx.x//cuda.warpsize

	val = __warp_max_reduce(val, size)

	#laneid = position of thread in warp
	if cuda.laneid == 0 and cuda.grid(1) < size:
		shared[s_n] = val

	cuda.syncthreads()

	val = np.nan
	if cuda.threadIdx.x < (cuda.blockDim.x + cuda.warpsize - 1)//cuda.warpsize:
		val = shared[cuda.laneid]

	if s_n == 0:
		val = __warp_max_reduce(val, size)

	return val

@cuda.jit('void(f8[:], f8[:], i8)')
def __max(_in, _out, size):
	start  = cuda.grid(1)
	stride = cuda.gridsize(1)
	# size   = _in.size
	# math.isnan()

	s_shape = 32
	shared  = cuda.shared.array(shape = s_shape, dtype = float64)

	#the frist thread in each block fills shared memory
	if start%cuda.blockDim.x == 0:
		for i in range(s_shape):
			shared[i] = np.nan #-np.inf

	#find mimumum along strided array
	val = _in[start]
	for n in range(start, size, stride):
		val = max(val, _in[n])

	val = __block_max_reduce(val, size, shared)
	if start%cuda.blockDim.x == 0:
		_out[cuda.blockIdx.x] = max(val, _out[cuda.blockIdx.x])

def max_gpu(d_in, size = None, stream = 0):
	if size is None:
		size = d_in.size
	size = min(size, d_in.size)

	threads = min(rthreads, size)
	blocks  = np.int64(min((size + threads - 1)//threads, 1024))

	#create flattened copy of input
	if len(d_in.shape) > 1:
		d_in = cuda.devicearray.DeviceNDArray(shape = size,
											  strides = (d_in.dtype.itemsize,),
											  dtype = d_in.dtype,
											  gpu_data = d_in.gpu_data)
		
	d_out = cuda.to_device(np.full(blocks, np.nan, dtype = np.float64))

	__max[blocks, threads](d_in, d_out, np.int64(size))

	__max[1, blocks](d_out, d_out, blocks)
	val = d_out[0:1]

	return val.copy_to_host()[0]
