import numpy as np
import bitarray as ba


class CyclicCodes:

    def __init__(self):
        self.n = 7
        self.k = 4
        self.gen = ba.bitarray('1011')


    def encode(self, input):
        out = ba.bitarray('0000000')
        tmp_gen = self.gen
        if self.k != len(input):
            print('invalid length input')
            pass

        for i in range(0, self.k):
            if input[i] == 1:
                out[i:i+4] ^= tmp_gen

        return out


    def remainder(self, input):
        remaind = ba.bitarray(input)
        tmp_gen = ba.bitarray('0001011')
        length_input = len(input)
        for i in range(self.n - 1, self.n - self.k + 1, -1):
            if remaind[i] == 1:
                remaind[i:i - self.k] ^= tmp_gen

        return remaind



cc = CyclicCodes()
input1 = ba.bitarray('1010')
encode = cc.encode(input1)
print(encode)
input2 = ba.bitarray('0001101')
remaind = cc.remainder(input2)
print(remaind)
