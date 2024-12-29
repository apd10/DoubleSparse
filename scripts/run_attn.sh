#!/bin/bash

bszs=(1)
#bszs=(1 4 8 16 32)
#ctxs=(2048 4096 8192 16384)
ctxs=(16384 32768 65536 131072)

for bsz in "${bszs[@]}"; do
    for ctx in "${ctxs[@]}"; do
        python3 ../models/triton_kernels/torchatt.py $bsz $ctx
    done
done
