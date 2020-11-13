import bitarray as ba
import numpy as np


class GolayCode:

    def __init__(self):
        self.k = 12
        self.n = 24
        self.B = (
            ba.bitarray('110111000101'),
            ba.bitarray('101110001011'),
            ba.bitarray('011100010111'),
            ba.bitarray('111000101101'),
            ba.bitarray('110001011011'),
            ba.bitarray('100010110111'),
            ba.bitarray('000101101111'),
            ba.bitarray('001011011101'),
            ba.bitarray('010110111001'),
            ba.bitarray('101101110001'),
            ba.bitarray('011011100011'),
            ba.bitarray('111111111110')
        )

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
        G = np.zeros((self.k, self.n), dtype=bool)
        for i in range(0, self.k):
            G[i][i] = True
            b_i = self.B[i].tolist()
            for j in range(self.n - self.k):
                G[i][self.k + j] = b_i[j]
        # np_G = np.ndarray((G[i].tolist() for i in range(self.k)), dtype=bool)
        np_input = np.array(input.tolist(), dtype=bool)
        encode = np.dot(np_input, G)
        return self.list2bitarray(encode)

    def decode(self, input_n):
        H = np.zeros((self.n, self.k), dtype=bool)
        for i in range(0, self.k):
            H[i][i] = True
            H[self.k + i] = self.B[i].tolist()
        np_input = np.array(input_n.tolist(), dtype=bool)
        s = np.dot(np_input, H)
        wt_s = self.wt(s)

        if wt_s <= 3:
            ba_u = ba.bitarray('0'*self.n)
            for i in range(self.k):
                ba_u[i] = s[i]
            return input_n ^ ba_u

        ba_s = self.list2bitarray(s)
        for i in range(self.k):
            tmp = ba_s ^ self.B[i]
            if self.wt(tmp) <= 2:
                ba_u = ba.bitarray('0'*self.n)
                ba_u[self.k + i] = True
                for j in range(self.k):
                    ba_u[j] = tmp[j]
                return input_n ^ ba_u

        np_B = np.zeros((self.k, self.k), dtype=bool)
        for i in range(self.k):
            np_B[i] = self.B[i].tolist()
        sB = np.dot(s, np_B)
        ba_sB = self.list2bitarray(sB)
        wt_sB = self.wt(sB)
        if wt_sB <= 3:
            ba_u = ba.bitarray('0' * self.n)
            for i in range(self.n - self.k):
                ba_u[self.n - self.k + i] = s[i]
            return input_n ^ ba_u

        for i in range(self.k):
            tmp = ba_sB ^ self.B[i]
            if self.wt(tmp) <= 2:
                ba_u = ba.bitarray('0' * self.n)
                ba_u[i] = True
                for j in range(self.n - self.k):
                    ba_u[self.k + j] = tmp[j]
                return input_n ^ ba_u

        print("WARNING! The error cannot be corrected, please re-enter the input signal.")
        return None


gc = GolayCode()
input_k = ba.bitarray('011011111010')
encode = gc.encode(input_k)
print(f"Encode == {encode}")
index_error = 1
encode[index_error] = encode[index_error] is False
print(f"Encode with an error in the {index_error} bit: {encode}")
decode = gc.decode(encode)
if decode is None:
   print('Decoding failed')
else:
   print(f'Decode == {decode}')
