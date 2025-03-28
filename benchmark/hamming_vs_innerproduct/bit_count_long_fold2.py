import torch
import cupy as cp
import numpy as np
from math import sqrt

FOLD=8

bit_count_kernel_long = cp.RawKernel(r'''
extern "C" __global__
void bit_count_kernel_long(const unsigned long long int* input, unsigned long long int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __popcll(input[idx]) + __popcll(input[idx+size]) + __popcll(input[idx+2*size]) + __popcll(input[idx+3*size]) + __popcll(input[idx+4*size]) + __popcll(input[idx+5*size]) + __popcll(input[idx+6*size]) + __popcll(input[idx+7*size]) ;
    }
}
''', 'bit_count_kernel_long')


def gpu_bit_count_long(input_tensor, output_tensor, fold=2):
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    if input_tensor.dtype != torch.int64:
        raise ValueError("Input tensor must be of type torch.int64 (long)")
    
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.cuda()
    
    input_cp = cp.from_dlpack(input_tensor)
    output_cp = cp.from_dlpack(output_tensor)
    
    total_elements = output_cp.size
    threads_per_block = 256
    blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
    
    bit_count_kernel_long((blocks_per_grid,), (threads_per_block,),
                          (input_cp, output_cp, total_elements))


def latency():
    for kvsize in 2**np.arange(1, 1+26+int(sqrt(FOLD))):
        if kvsize//FOLD < 2:
            continue
        k = torch.randint(2**63-1, (kvsize, 1), dtype=torch.int64, device="cuda:0")
        q = torch.randint(2**63-1, (1,1), dtype=torch.int64, device="cuda:0")
        ans = torch.zeros(kvsize//FOLD,1, dtype=torch.int64, device="cuda:0")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(100):
            gpu_bit_count_long(torch.bitwise_xor(k, q), ans)
        end.record()
        end.synchronize()
        elapsed_time = start.elapsed_time(end)
        print(f"Size:{kvsize//FOLD} time: {(elapsed_time/100):.3f} ms")


if __name__ == "__main__":
    latency()
