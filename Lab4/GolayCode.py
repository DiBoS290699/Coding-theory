import bitarray as ba
import numpy as np


class GolayCode:

    def __init__(self):
        self.k = 12
        self.n = 24
        self.B = np.array([
            [True, True, False, True, True, True, False, False, False, True, False, True],
            [True, False, True, True, True, False, False, False, True, False, True, True],
            [False, True, True, True, False, False, False, True, False, True, True, True],
            [True, True, True, False, False, False, True, False, True, True, False, True],
            [True, True, False, False, False, True, False, True, True, False, True, True],
            [True, False, False, False, True, False, True, True, False, True, True, True],
            [False, False, False, True, False, True, True, False, True, True, True, True],
            [False, False, True, False, True, True, False, True, True, True, False, True],
            [False, True, False, True, True, False, True, True, True, False, False, True],
            [True, False, True, True, False, True, True, True, False, False, False, True],
            [False, True, True, False, True, True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True, True, True, True, True, False]
        ])

    def wt(self, input_signal):
        wt = 0
        for i in range(len(input_signal)):
            if input_signal[i]:
                wt += 1
        return wt

    def list2bitarray(self, list):
        ba_list = ba.bitarray('0'*len(list))
        for k in range(len(list)):
            ba_list[k] = list[k]
        return ba_list

    def encode(self, input):
        if len(input) != self.k:
            print("Error!")
            return None
        I = np.eye(self.k, dtype=bool)
        G = np.concatenate([I, self.B], axis=1)
        if input.__class__ == ba.bitarray().__class__:
            np_input = np.array(input.tolist(), dtype=bool)
        else:
            np_input = np.array(input, dtype=bool)
        encode = np.dot(np_input, G)
        return self.list2bitarray(encode)

    def decode(self, input_n):
        I = np.eye(self.k, dtype=bool)
        H = np.concatenate([I, self.B])

        if input_n.__class__ == ba.bitarray().__class__:
            np_input = np.array(input_n.tolist(), dtype=bool)
        else:
            np_input = np.array(input_n, dtype=bool)
        s = np.dot(np_input, H)                 # Step 1
        wt_s = self.wt(s)

        if wt_s <= 3:                           # Step 2
            u = np.zeros(self.n)
            for i in range(self.k):
                u[i] = s[i]
            return np.logical_xor(np_input, u)

        # ba_s = self.list2bitarray(s)
        for i in range(self.k):                 # Step 3
            tmp = np.logical_xor(s, self.B[i])
            if self.wt(tmp) <= 2:
                u = np.zeros(self.n)
                u[self.k + i] = True
                for j in range(self.k):
                    u[j] = tmp[j]
                return np.logical_xor(np_input, u)

        sB = np.dot(s, self.B)                  # Step 4
        wt_sB = self.wt(sB)
        if wt_sB <= 3:                          # Step 5
            u = np.zeros(self.n)
            for i in range(self.n - self.k):
                u[self.n - self.k + i] = s[i]
            return np.logical_xor(np_input, u)

        for i in range(self.k):                 # Step 6
            tmp = np.logical_xor(sB, self.B[i])
            if self.wt(tmp) <= 2:
                u = np.zeros(self.n)
                u[i] = True
                for j in range(self.n - self.k):
                    u[self.k + j] = tmp[j]
                return np.logical_xor(np_input, u)
                                                # Step 7
        print("WARNING! The error cannot be corrected, please re-enter the input signal.")
        return None


# gc = GolayCode()
# input_k = ba.bitarray('011011111010')
# encode = gc.encode(input_k)
# print(f"Encode == {encode}")
# index_error = 1
# encode[index_error] = not encode[index_error]
# print(f"Encode with an error in the {index_error} bit: 101111101111010010010010")
# decode = gc.decode(ba.bitarray('101111101111010010010010'))
# if decode is None:
#    print('Decoding failed')
# else:
#    print(f'Decode == {decode}')

