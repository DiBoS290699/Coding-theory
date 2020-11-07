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
        G = np.zeros((12, 24), dtype=bool)
        for i in range(0, self.k):
            G[i][i] = True
            G[i][self.k:] = self.B[i].tolist()
        # np_G = np.ndarray((G[i].tolist() for i in range(self.k)), dtype=bool)
        np_input = np.array(input.tolist(), dtype=bool)
        encode = np.dot(np_input, G)
        return self.list2bitarray(encode)

    def decode(self, input_n):
        H = np.zeros((24, 12), dtype=bool)
        for i in range(0, self.k):
            H[i][i] = True
            H[self.k + i] = self.B[i].tolist()
        np_input = np.array(input_n.tolist(), dtype=bool)
        s = np.dot(np_input, H)
        wt_s = self.wt(s)
        u = []
        if wt_s <= 3:
            u = np.ndarray(buffer=(s, np.zeros((1, self.k), dtype=bool)), shape=(1, 1))
            ba_u = self.list2bitarray(u)
            return input_n ^ ba_u
        for i in range(self.k):
            tmp = s ^ self.B[i]
            if self.wt(tmp) <= 2:
                e_i = np.zeros((1, 12), dtype=bool)
                e_i[i] = True
                u = np.ndarray(buffer=(tmp, e_i), shape=(1, 1))
                ba_u = self.list2bitarray(u)
                return input_n ^ ba_u
        s_B =


gc = GolayCode()
input_k = ba.bitarray('011010100010')
encode = gc.encode(input_k)
print(f"Encode == {encode}")
print(True + True)
