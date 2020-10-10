import numpy as np
import bitarray as ba


class CyclicCodes:

    def __init__(self):
        self.n = 7
        self.k = 4
        self.t = 1
        self.gen = ba.bitarray('1011')


    def task_2(self, input):
        out = ba.bitarray('0000000')
        tmp_gen = self.gen
        if self.k != len(input):
            print('invalid length input')
            pass

        for i in range(0, self.k):
            if input[i] == 1:
                out[i:i+4] ^= tmp_gen

        return out


    def task_3(self, input):
        remaind = ba.bitarray(input)
        for i in range(self.n - 1, self.n - self.k - 1, -1):
            if remaind[i] == 1:
                remaind[i - (self.n - self.k): i + 1] ^= self.gen

        return remaind


    def task_4(self, input):
        new_input = ba.bitarray('0000000')
        new_input[self.n - self.k:] = input
        rem = self.task_3(new_input)
        new_input[:self.n - self.k] = rem[:self.n - self.k]
        return new_input


    def task_5(self):
        syndromes = {}
        codes = []
        for i in range(0, 2**self.n):
            codes.append(ba.bitarray(bin(i)[2:].zfill(self.n)))
        for err in codes:
            wt = 0
            for i in err:
                wt += int(i)
            if wt <= self.t:
                rem = self.task_3(err)
                syndromes[rem.to01()] = err
        return syndromes



cc = CyclicCodes()
input1 = ba.bitarray('1010')
encode = cc.task_2(input1)
print(f'encode: {encode}')
input2 = ba.bitarray('0001101')
remaind = cc.task_3(input2)
print(f'Remaind {input2}: {remaind}')
cod = cc.task_4(input1)
print(f'codyng {input1}: {cod}')
syndromes = cc.task_5()
print(f'syndromes: {syndromes}')
