import numpy as np
import bitarray as ba


class CyclicCodes:

    def __init__(self):
        self.n = 15
        self.k = 7
        self.t = 2
        self.gen = ba.bitarray('100010111')

    def encode(self, input):
        if len(input) != self.k:
            print("ERROR! Invalid code length.")
            return None
        out = ba.bitarray('0'*self.n)
        tmp_gen = self.gen
        for i in range(0, self.k):
            if input[i] == 1:
                out[i:i + len(tmp_gen)] ^= tmp_gen
        return out

    def remainder(self, input):
        if len(input) != self.n:
            print("ERROR! Invalid code length.")
            return None
        remaind = ba.bitarray(input)
        for i in range(self.n - 1, self.n - self.k - 1, -1):
            if remaind[i] == 1:
                remaind[i - (self.n - self.k): i + 1] ^= self.gen
        return remaind

    def encode_sys(self, input):
        if len(input) != self.k:
            print("ERROR! Invalid code length.")
            return None
        new_input = ba.bitarray('0'*self.n)
        new_input[self.n - self.k:] = input
        rem = self.remainder(new_input)
        new_input[:self.n - self.k] = rem[:self.n - self.k]
        return new_input

    def make_table(self):
        syndromes = {}
        codes = []
        for i in range(0, 2 ** self.n):
            codes.append(ba.bitarray(bin(i)[2:].zfill(self.n)))
        for err in codes:
            wt = 0
            for i in err:
                wt += int(i)
            if wt <= self.t:
                rem = self.remainder(err)
                syndromes[rem.to01()] = err
        return syndromes

    def make_a_mistake(self, code):
        if len(code) != self.n:
            print("ERROR! Invalid code length.")
            return code
        make_a_mistake = np.random.randint(0, 2)
        if make_a_mistake == 0:
            return code
        else:
            error_bit = np.random.randint(0, self.n)
            code[error_bit] = 0 if code[error_bit] == 1 else 0
            return code

    def search_error(self, path_to_file):
        file_bitarray = ba.bitarray()
        syndromes = self.make_table()
        errors = {}
        with open(path_to_file, 'rb') as file:
            file_bitarray.fromfile(file)
        for i in range(0, len(file_bitarray), self.n):
            code = file_bitarray[i: i+self.n]
            if len(code) != self.n:
                continue
            code = self.make_a_mistake(code)
            file_bitarray[i: i+self.n] = code
            syndrome = syndromes.get(file_bitarray[i: i+self.n].to01())
            if syndrome is not None:
                errors[i] = syndrome
            else:
                continue
        # ["path", "txt"]
        path_to_file_arr = path_to_file.split(".")
        with open(path_to_file_arr[0] + "_decode." + path_to_file_arr[1], 'wb') as f:
            file_bitarray.tofile(f)
        return errors


cc = CyclicCodes()
with open("Hello.txt", 'wb') as file:
    file_bitarray = ba.bitarray('0'*cc.n*225)
    for i in range(0, len(file_bitarray)):
        file_bitarray[i] = np.random.randint(0, 2) == 1
    file_bitarray.tofile(file)
input1 = ba.bitarray('1010001')
encode = cc.encode(input1)
print(f'encode: {encode}')
input2 = ba.bitarray('100100110111101')
remaind = cc.remainder(input2)
print(f'Remaind {input2}: {remaind}')
cod = cc.encode_sys(input1)
print(f'coding {input1}: {cod}')
syndromes = cc.make_table()
print(f'syndromes: {syndromes}')
errors = cc.search_error("Hello.txt")
print(f"Errors: {errors}")
