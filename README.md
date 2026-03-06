# peped - Tensor, Autograd and Modern ML Library


A C++ tensor and automatic differentiation library with a Python interface via pybind11. Built from scratch with no noob-ass tech like — PyTorch ,TensorFlow, straigh up masochism for the boys back at home. Implements stride-based N-dimensional tensors, reverse-mode autograd with topological sort, BLAS-accelerated matrix multiplication, and OpenMP parallelism, Neural network model development and optimizations.

---
## Features
- **Stride-based tensors** — zero-copy views, transpose, permute, slice, and broadcast via stride manipulation. Non-contiguous memory layouts are fully supported.
- **Reverse-mode autograd** — computation graph built dynamically during the forward pass, gradients computed via topological sort on the backward pass.
- **BLAS integration** — matrix multiplication dispatches to `cblas_dgemm` for single, batched, and broadcast cases.
- **OpenMP parallelism** — elementwise operations and activations parallelised across CPU cores with `simd` vectorisation hints.
- **Zero-copy numpy interop** — tensors expose the Python buffer protocol, so `np.array(tensor)` and `Tensor.from_numpy(arr)` share memory without copying.
- **Full neural network layer support** — LSTM, Multi-Head Attention, GRN, and Temporal Fusion Transformer implemented on top of the library.
---



## Requirements
- C++20
- CMake >= 3.15
- OpenBLAS or any CBLAS-compatible library
- OpenMP
- Python >= 3.10
- pybind11

---



## Build

```bash

git clone https://github.com/joelkurien/tftcore
cd tftcore

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

```



This produces `peped.cpython-3XX-x86_64-linux-gnu.so` in the `build/` directory.



To run Python from the build directory:

```bash

cd build
python3
>>> import peped

```

To install system-wide into your conda environment:

```bash
pip install --no-build-isolation -e .
```

---

## Module structure

```

peped
├── peped.tensor          # Raw tensor ops, no gradient tracking
│   └── Tensor            # Core tensor class
└── peped.autograd        # Gradient-tracked ops
│   └── TensorX           # Autograd node wrapping Tensor
└── peped.ease_tensor     # Factory helpers for TensorX creation
```



---

## Quick start

### Tensor (no autograd)

```python

import peped
import numpy as np

# Construction
t = peped.tensor.Tensor([3, 4])          # zero tensor, shape [3, 4]
t = peped.tensor.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])   # from data
t = peped.tensor.Tensor.from_numpy(np.random.randn(3, 4)) # from numpy

# numpy interop — zero copy

arr = t.to_ndarray()   # returns np.array sharing memory with t
arr = np.array(t)      # same via buffer protocol

# Shape ops — zero copy views

t.reshape([4, 3])
t.transpose()                    # swap last two axes
t.transpose(0, 2)                # swap axes 0 and 2
t.permute([2, 0, 1])
t.unsqueeze(0)                   # insert dim at axis 0
t.squeeze()                      # remove all size-1 dims
t.expand([6, 4])                 # broadcast to target shape
t.slice([0, 0], [2, 2])          # view of first 2x2 block
t.chunk(4, axis=1)               # split into 4 chunks along axis 1

# Indexing

val = t.at([1, 2])               # read element
t.put([1, 2], 99.0)              # write element

# Arithmetic

a + b    # elementwise add
a - b    # elementwise subtract
a * b    # elementwise multiply
a / b    # elementwise divide

a + 2.0  # scalar broadcast
2.0 * a  # scalar broadcast

# Reductions

t.sum(axis=1)
t.mean(axis=0)
t.maximum(axis=1)
t.minimum(axis=0)

# Math

t.sqrt()
t.log()
t.exp()
t.pow(2.0)

# Activations

t.relu()
t.gelu()
t.sigmoid()
t.tanh()
t.elu(alpha=1.0)
t.softmax(axis=1)
t.log_softmax(axis=1)
t.layer_norm(gamma, beta, axis=1)

# Initialisation

w = peped.tensor.Tensor([512, 256])
w.xavier_ud(fan_in=512, fan_out=256)

# Dropout — returns (result, mask) tuple
result, mask = t.dropout(p=0.1, training=True)

# Free functions

peped.tensor.matmul(a, b)
peped.tensor.concatenate([a, b, c], axis=0)
peped.tensor.ones([3, 3])
peped.tensor.dot(x, y, axis=1)
peped.tensor.elemental_max(a, b)

```

---

### TensorX (with autograd)

```python

import peped
import numpy as np

ag = peped.autograd

# Create autograd tensors

data = peped.tensor.Tensor.from_numpy(np.random.randn(4, 8))
x = ag.TensorX(data, requires_grad=True)

# Or use factory helpers

x = peped.ease_tensor.deep_create([4, 8], requires_grad=True)
x = peped.ease_tensor.deep_create([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)

# Forward pass — builds computation graph automatically

y = ag.relu(x)
z = ag.matmul(y, ag.TensorX(peped.tensor.ones([8, 4]), requires_grad=False))
loss = ag.mean(z, axis=1)

loss = ag.mean(loss, axis=0)

# Backward pass

loss.backward()

# Access gradients

grad = x.nd_grad()          # returns np.array
print(grad.shape)            # (4, 8)

# Zero gradients before next step

x.zero_grad()

# Inspect data

print(x.shape())             # [4, 8]
print(x.size())              # 32
print(x)                     # TensorX(shape=(4, 8), requires_grad=True)

# All autograd ops

ag.add(x, y)                 # x + y
ag.add(x, 2.0)               # x + scalar
ag.subtract(x, y)
ag.multiply(x, y)
ag.divide(x, y)
ag.sqrt(x)
ag.log(x)
ag.exp(x)
ag.pow(x, 2.0)
ag.sum(x, axis=1)
ag.mean(x, axis=0)
ag.var(x, axis=1)
ag.maximum(x, axis=0)
ag.minimum(x, axis=1)
ag.relu(x)
ag.gelu(x)
ag.sigmoid(x)
ag.tanh(x)
ag.elu(x, alpha=1.0)
ag.softmax(x, axis=1)
ag.log_softmax(x, axis=1)
ag.layer_norm(x, gamma, beta, axis=1)
ag.glu(x, axis=1)
ag.reGlu(x)
ag.matmul(x, y)
ag.transpose(x)
ag.permute(x, axes=[1, 0])
ag.reshape(x, [8, 4])
ag.squeeze(x)
ag.unsqueeze(x, axis=0)
ag.expand(x, [8, 8])
ag.chunk(x, num_chunks=4, axis=1)
ag.concat([x, y], axis=0)
ag.stack([x, y], axis=0)
ag.slice(x, start=[0, 0], shape=[2, 4])
ag.fill_mask(x, mask, replace=0.0)
ag.pinball_loss(y_true, y_pred, tau=0.5)

# Dropout — returns (TensorX result, Tensor mask)
result, mask = ag.dropout(x, p=0.1, training=True)

```

---

## Design notes

**Why stride-based views?** Transpose, permute, and slice return views into the same memory by manipulating strides rather than copying data. This matches numpy's internal design and avoids unnecessary allocation during forward passes.

**Why reverse-mode autograd?** For neural networks where the number of outputs (loss = scalar) is much smaller than the number of inputs (parameters), reverse-mode is O(1) backward passes vs O(n) for forward-mode. Each op registers a backward closure during the forward pass; `backward()` executes them in reverse topological order. Also for some reason there are no research papers regarding implementation process using traditional chain differentiation so I just settled on doing it using topological sort approach, not that anyone asked.

**Why BLAS for matmul?** Truthfully, I was lazy to implement the matrix multiplication from scratch and optimize it. Don't get me wrong I have tried to be innovative about it and implement a system where based on an estimated computation cost I would determine if i need to process the multiplication using CPU or a GPU ( a very cool concept ), but it was too complex at the time, so I gave up. Therefore I used BLAS. A more professional answer would be - `cblas_dgemm` is highly optimised for the target architecture and uses cache-blocking and SIMD internally. Writing a competitive matmul from scratch is non-trivial; BLAS handles it correctly for all matrix layouts including transposed operands.

**Why OpenMP + simd?** Elementwise ops like `relu`, `sigmoid`, and `gelu` are embarrassingly parallel. `#pragma omp parallel for simd simdlen(4)` distributes work across CPU cores while the `simd` clause hints to the compiler to use AVX2 SIMD lanes (4 doubles per register with `-mavx2`). 

## Future Implementations  

**Better Matmul** Develop the matrix multiplication process to not use BLAS, but rather use the system where based on the size of the operands we decide whether to use a CPU or a GPU to do the matrix multiplication

**GPU usage** Currently all the operations use extensive CPU parallelizations, but later I want to build a library, where the user can use the same functions but choose to use a CPU computation or a GPU computation based on their requirements. Something similar to the `try_gpu()` function in **Pytorch**.

**Machine Learning usage** I want to build a proper ML framework using my library with a wide range of ML layers, models, search algorithms, initializers and make it more than a side project.

**Optimizations** Currently the tensor module of the library is dependent on a single pointer, giving it a high risk of causing dangling pointer. I want to remove it using smart pointers like `shared_ptr` and `unique_ptr`. But to do this I will have to rethink my current implementations. I also want to make my elementwise calculations more parallelized and low level for faster calculation. I also want to directly use the ALU for faster CPU computation.
