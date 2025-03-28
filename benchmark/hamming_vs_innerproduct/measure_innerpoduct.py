import torch
import numpy as np

for kvsize in 2**np.arange(26):
    k = torch.rand(kvsize, 128, dtype=torch.float16, device="cuda:0")
    q = torch.rand(1,128, dtype=torch.float16, device="cuda:0")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(100):
        ans = torch.matmul(k, q.T)
    end.record()
    end.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"Size:{kvsize} time: {(elapsed_time/100):.3f} ms")

