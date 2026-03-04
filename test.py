"""
peped library demo
------------------
Demonstrates the core features of the C++ tensor library
exposed to Python via pybind11.
"""

import peped
import numpy as np

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def check(label, result, expected=None):
    status = "✓" if expected is None else ("✓" if result == expected else "✗")
    print(f"  [{status}] {label}: {result}")


# ─────────────────────────────────────────────────────────
# 1. Construction
# ─────────────────────────────────────────────────────────
section("1. Tensor Construction")

# From shape (zero initialised)
t1 = peped.Tensor([3, 4])
check("shape", t1.shape(), [3, 4])
check("ndim",  t1.ndim(),  2)
check("size",  t1.size(),  12)

# From flat data + shape
t2 = peped.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
check("data+shape construction", t2.shape(), [2, 3])

# From numpy

arr = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
t3 = peped.Tensor.from_numpy(arr)
check("from_numpy shape", t3.shape(), [2, 3])



# ─────────────────────────────────────────────────────────
# 2. Numpy interop
# ─────────────────────────────────────────────────────────
section("2. Numpy Interop (zero-copy)")

np_view = t3.to_ndarray()

check("to_ndarray shape",  tuple(np_view.shape), (2, 3))
check("to_ndarray dtype",  str(np_view.dtype),   "float64")
check("values match",      np_view[0, 0],         1.0)
check("values match",      np_view[1, 2],         6.0)

# numpy array() also works via buffer protocol

np_buf = np.array(t3)
check("buffer protocol shape", tuple(np_buf.shape), (2, 3))


# ─────────────────────────────────────────────────────────
# 3. Indexing
# ─────────────────────────────────────────────────────────
section("3. Indexing")

check("at([0,0])", t2.at([0, 0]), 1.0)
check("at([1,2])", t2.at([1, 2]), 6.0)

t2.put([0, 0], 99.0)
check("put([0,0], 99)", t2.at([0, 0]), 99.0)
t2.put([0, 0], 1.0)   # reset


# ─────────────────────────────────────────────────────────
# 4. Shape operations (zero-copy views)
# ─────────────────────────────────────────────────────────
section("4. Shape Operations")

reshaped = t2.reshape([3, 2])
check("reshape [2,3] -> [3,2]", reshaped.shape(), [3, 2])

transposed = t2.transpose()
check("transpose shape", transposed.shape(), [3, 2])

unsqueezed = t2.unsqueeze(0)
check("unsqueeze(0)", unsqueezed.shape(), [1, 2, 3])

squeezed = unsqueezed.squeeze(0)
check("squeeze(0)", squeezed.shape(), [2, 3])

permuted = peped.Tensor([1.0]*24, [2, 3, 4]).permute([2, 0, 1])

check("permute [2,3,4] -> [4,2,3]", permuted.shape(), [4, 2, 3])

check("is_contiguous (original)", t2.is_contiguous(), True)
check("is_contiguous (transposed)", transposed.is_contiguous(), False)
check("to_contiguous", transposed.to_contiguous().is_contiguous(), True)


# ─────────────────────────────────────────────────────────
# 5. Arithmetic operators
# ─────────────────────────────────────────────────────────
section("5. Arithmetic Operators")

a = peped.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = peped.Tensor([10.0, 20.0, 30.0, 40.0], [2, 2])


c = a + b
check("tensor + tensor [0,0]", c.at([0, 0]), 11.0)

c = a + 5.0
check("tensor + scalar [0,1]", c.at([0, 1]), 7.0)

c = 3.0 + a        # tests __radd__
check("scalar + tensor [1,0]", c.at([1, 0]), 6.0)

c = b - a
check("tensor - tensor [1,1]", c.at([1, 1]), 36.0)

c = a * b
check("tensor * tensor [0,0]", c.at([0, 0]), 10.0)

c = 2.0 * a        # tests __rmul__
check("scalar * tensor [0,1]", c.at([0, 1]), 4.0)

c = b / a
check("tensor / tensor [0,0]", c.at([0, 0]), 10.0)



# ─────────────────────────────────────────────────────────
# 6. Reductions
# ─────────────────────────────────────────────────────────
section("6. Reductions")

t = peped.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

s = t.sum(1)
check("sum(axis=1) shape",  s.shape(), [2, 1])
check("sum(axis=1) row 0",  s.at([0, 0]), 6.0)   # 1+2+3
check("sum(axis=1) row 1",  s.at([1, 0]), 15.0)  # 4+5+6

m = t.mean(0)
check("mean(axis=0) shape", m.shape(), [1, 3])
check("mean(axis=0) col 0", m.at([0, 0]), 2.5)   # (1+4)/2


# ─────────────────────────────────────────────────────────

# 7. Math functions

# ─────────────────────────────────────────────────────────
section("7. Math Functions")

t = peped.Tensor([1.0, 4.0, 9.0, 16.0], [2, 2])

sq = t.sqrt()
check("sqrt [0,0]", sq.at([0, 0]), 1.0)
check("sqrt [0,1]", sq.at([0, 1]), 2.0)

lg = t.log()
check("log [0,0] ≈ 0", round(lg.at([0, 0]), 4), 0.0)

ex = peped.Tensor([0.0, 1.0, 0.0, 1.0], [2, 2]).exp()
check("exp(0) = 1", round(ex.at([0, 0]), 4), 1.0)
check("exp(1) ≈ 2.7183", round(ex.at([0, 1]), 4), 2.7183)

pw = peped.Tensor([2.0, 3.0, 4.0, 5.0], [2, 2]).pow(2)
check("pow(2) [0,0]", pw.at([0, 0]), 4.0)
check("pow(2) [0,1]", pw.at([0, 1]), 9.0)


# ─────────────────────────────────────────────────────────
# 8. Activations
# ─────────────────────────────────────────────────────────
section("8. Activation Functions")


t = peped.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [2, 3])


relu_t = t.relu()
check("relu(-2) = 0",  relu_t.at([0, 0]), 0.0)
check("relu(2)  = 2",  relu_t.at([1, 1]), 2.0)

sig_t = t.sigmoid()
check("sigmoid(0) = 0.5", round(sig_t.at([0, 2]), 4), 0.5)

tanh_t = t.tanh()
check("tanh(0) = 0.0",   round(tanh_t.at([0, 2]), 4), 0.0)

# softmax rows should sum to 1
sm = t.softmax(1)
row0_sum = sum(sm.at([0, j]) for j in range(3))
check("softmax row sum ≈ 1", round(row0_sum, 6), 1.0)

gelu_t = t.gelu()
check("gelu(-2) ≈ 0 (negative suppressed)", gelu_t.at([0, 0]) < 0.0, True)
check("gelu(3)  ≈ 3 (positive preserved)",  gelu_t.at([1, 2]) > 2.5, True)



# ─────────────────────────────────────────────────────────
# 9. Matmul
# ─────────────────────────────────────────────────────────
section("9. Matrix Multiplication (BLAS)")

a = peped.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])  # [[1,2],[3,4]]
b = peped.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])  # identity

c = peped.matmul(a, b)
check("matmul with identity [0,0]", c.at([0, 0]), 1.0)
check("matmul with identity [1,1]", c.at([1, 1]), 4.0)


# batch matmul
ba = peped.Tensor(list(range(1, 9)),  [2, 2, 2])
bb = peped.Tensor([1.0]*8,            [2, 2, 2])
bc = peped.matmul(ba, bb)
check("batch matmul shape", bc.shape(), [2, 2, 2])


# ─────────────────────────────────────────────────────────
# 10. Chunking and concatenation
# ─────────────────────────────────────────────────────────
section("10. Chunk and Concatenate")


t = peped.Tensor(list(range(1, 13)), [2, 6])
chunks = t.chunk(3, 1)          # split axis=1 into 3 chunks
check("chunk count",          len(chunks), 3)
check("chunk shape",          chunks[0].shape(), [2, 2])

cat = peped.concatenate(chunks, 1)
check("concatenate shape",    cat.shape(), [2, 6])
check("concatenate value",    cat.at([0, 0]), 1.0)


# ─────────────────────────────────────────────────────────
# 11. Xavier initialisation
# ─────────────────────────────────────────────────────────
section("11. Xavier Initialisation")


w = peped.Tensor([4, 8])
w.xavier_ud(4, 8)

arr = np.array(w)
check("xavier values in range [-1, 1]", bool(np.all(np.abs(arr) <= 1.0)), True)
check("xavier not all zero",            bool(np.any(arr != 0.0)),          True)


# ─────────────────────────────────────────────────────────
# 12. Free functions
# ─────────────────────────────────────────────────────────
section("12. Free Functions")

ones = peped.ones([3, 3])
check("ones shape",   ones.shape(), [3, 3])
check("ones value",   ones.at([1, 1]), 1.0)

a = peped.Tensor([-1.0, 2.0, -3.0, 4.0], [2, 2])
b = peped.Tensor([1.0, -2.0, 3.0, -4.0], [2, 2])
em = peped.elemental_max(a, b)
check("elemental_max [0,0]", em.at([0, 0]), 1.0)
check("elemental_max [1,1]", em.at([1, 1]), 4.0)


# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  Demo complete")
print(f"{'='*50}\n")
