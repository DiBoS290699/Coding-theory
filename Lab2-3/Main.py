import numpy as np
import bitarray as ba


class CyclicCodes:

    def __init__(self, gen, n=7, k=4, t=1):
        self.n = n
        self.k = k
        self.t = t
        self.gen = ba.bitarray(gen)

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

    # Возвращение кода со случайной ошибкой
    def make_a_mistake(self, code):
        if len(code) != self.n:
            print("ERROR! Invalid code length.")
            return code
        # make_a_mistake = np.random.randint(0, 2)    # 1 - создать ошибку, 0 - не выполнять ошибку
        # if make_a_mistake == 0:
        #     return code
        # else:
        #     error_bit = np.random.randint(0, self.n)    # Индекс бита, в котором будет ошибка
        #     code[error_bit] = 0 if code[error_bit] == 1 else 0
        #     return code
        error_bit = np.random.randint(0, self.n)  # Индекс бита, в котором будет ошибка
        code[error_bit] = not code[error_bit]
        return code

    # Кодировка файла path_to_input_file в новый файл path_to_output_file.
    # Аргумент error == false - кодировка без помех
    # Аргумент error == true - кодировка с помехой в одном бите в каждом коде
    def sys_encode_file(self, path_to_input_file, path_to_output_file, error=False):
        file_bitarray = ba.bitarray()
        with open(path_to_input_file, 'rb') as file:
            file_bitarray.fromfile(file)        # Чтение файла
        if not error:
            for i in range(0, len(file_bitarray), self.n):
                file_bitarray[i: i + self.k] = self.encode_sys(file_bitarray[i: i + self.k])
        else:
            for i in range(0, len(file_bitarray), self.n):
                error_code = file_bitarray[i: i + self.k]
                error_code = self.encode_sys(error_code)
                error_code = self.make_a_mistake(error_code)
                file_bitarray[i: i + self.k] = error_code
        with open(path_to_output_file, 'wb') as f:
            file_bitarray.tofile(f)
        print(f"Encoding ({path_to_input_file} in {path_to_output_file}) completed successfully")

    # Декодирование файла path_to_input_file в файл path_to_output_file
    def sys_decode_file(self, path_to_input_file, path_to_output_file):
        file_bitarray = ba.bitarray()
        with open(path_to_input_file, 'rb') as file:
            file_bitarray.fromfile(file)        # Чтение файла
        syndromes = self.make_table()  # Возвращение таблицы синдромов класса
        for i in range(0, len(file_bitarray), self.k):
            code = file_bitarray[i: i + self.n]
            if len(code) != self.n:
                continue
            rem = self.remainder(code)
            err_vector = syndromes.get(rem.to01())
            if err_vector is None:
                print(f'ERROR! A syndrome ({rem.to01()}) from the table of syndromes was not found.')
                file_bitarray[i: i + self.n] = code[-self.k:]
            else:
                code ^= err_vector
                file_bitarray[i: i + self.n] = code[-self.k:]
        with open(path_to_output_file, 'wb') as f:
            file_bitarray.tofile(f)
        print(f"Decoding ({path_to_input_file} in {path_to_output_file}) completed successfully")


cc = CyclicCodes([1, 0, 0, 0, 1, 0, 1, 1, 1], n=15, k=7, t=2)
path_to_input_file = "Hello.txt"
path_to_output_file = "Hello_sys_encode.txt"
path_to_output_file_with_errors = "Hello_errors.txt"
path_to_output_file_without_errors = "Hello_without_errors.txt"
with open(path_to_input_file, 'wb') as file:
    file_bitarray = ba.bitarray('0'*cc.k*3)   # создание bitarray с определённым количеством нулей
    for i in range(0, len(file_bitarray)):
        file_bitarray[i] = np.random.randint(0, 2) == 1     # Заполнение bitarray рандомными битами
    file_bitarray.tofile(file)                  # Запись псевдослучайный bitarray в файл
input1 = ba.bitarray('1010110')
encode = cc.encode(input1)
print(f'encode: {encode}')
input2 = ba.bitarray('100100110010011')
remaind = cc.remainder(input2)
print(f'Remaind {input2}: {remaind}')
cod = cc.encode_sys(input1)
print(f'sys_encode {input1}: {cod}')
syndromes = cc.make_table()
print(f'syndromes: {syndromes}')
cc.sys_encode_file(path_to_input_file, path_to_output_file, error=False)
cc.sys_encode_file(path_to_input_file, path_to_output_file_with_errors, error=True)
cc.sys_decode_file(path_to_output_file_with_errors, path_to_output_file_without_errors)
