import bitarray as ba
import numpy as np


class RMCode:

    def __init__(self, r, m):
        self.r = None if r < 0 else r
        self.m = None if m < 0 or m < r else m
        self.k = None
        if self.r is not None and self.m is not None:
            self.k = 0
            for i in range(r + 1):
                self.k += np.math.factorial(m)/(np.math.factorial(i)*np.math.factorial(m - i))
            self.k = int(self.k)
        self.n = None if m is None else 2**m
        self.d = None if m is None or r is None else 2**(self.m - self.r)
        self.G_r_m = self.gen_matrix(r, m)

    def gen_matrix(self, r, m):
        if m is None or r is None or r < 0 or m < 0 or m < r or m == r == 0:
            print(f"ERROR! The value m or r is incorrect: m is {m}, r is {r}")
            return None
        elif r == 0:
            line = np.ones((1, 2**m), dtype=bool)
            return line
        elif r == m:
            bottom_line = np.zeros((1, 2**m), dtype=bool)
            bottom_line[0][-1] = True
            return np.concatenate([self.gen_matrix(m - 1, m), bottom_line])
        else:
            G_r_m_1 = np.concatenate([self.gen_matrix(r, m - 1), self.gen_matrix(r, m - 1)], axis=1)
            G_r_1_m_1 = self.gen_matrix(r - 1, m - 1)
            shape_zeros = np.shape(G_r_1_m_1)
            zeros = np.zeros(shape_zeros, dtype=bool)
            return np.concatenate([G_r_m_1, np.concatenate([zeros, G_r_1_m_1], axis=1)])

    def encode(self, input_k):
        if self.k != len(input_k):
            print(f'ERROR! The input_k has an incorrect size. The size must be equal to {self.k} instead of {len(input_k)}')
            return None
        if input_k.__class__ == ba.bitarray().__class__:
            np_input_k = np.array(input_k.tolist(), dtype=bool)
            return np.dot(np_input_k, self.G_r_m)
        else:
            return np.dot(input_k, self.G_r_m)


rmc = RMCode(1, 3)
# G_r_m = rmc.gen_matrix(3, 3)
# print(G_r_m)
input_k = []
for i in range(rmc.k):
    input_k.append(np.random.randint(0, 2, dtype=bool))
ba_input_k = ba.bitarray(input_k)
print(f"The input_k == {ba_input_k}")
encode = rmc.encode([False, False, True, True])
print(f"The encode == {encode}")
