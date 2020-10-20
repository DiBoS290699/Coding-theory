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

    # Возвращение кода со случайной ошибкой, либо без неё
    def make_a_mistake(self, code):
        if len(code) != self.n:
            print("ERROR! Invalid code length.")
            return code
        make_a_mistake = np.random.randint(0, 2)    # 1 - создать ошибку, 0 - не выполнять ошибку
        if make_a_mistake == 0:
            return code
        else:
            error_bit = np.random.randint(0, self.n)    # Индекс бита, в котором будет ошибка
            code[error_bit] = 0 if code[error_bit] == 1 else 0
            return code

    # Кодировка файла, обнаружение ошибки, декодировка файла
    def search_error(self, path_to_file):
        file_bitarray = ba.bitarray()
        syndromes = self.make_table()   # Возвращение таблицы синдромов класса
        errors = {}     # Словарь ошибок: key - индекс начала кода с ошибкой, значение - синдромкода с ошибкой
        with open(path_to_file, 'rb') as file:
            file_bitarray.fromfile(file)        # Чтение файла
        for i in range(0, len(file_bitarray), self.n):
            code = file_bitarray[i: i+self.n]
            if len(code) != self.n:
                continue
            code = self.make_a_mistake(code)    # Создание ошибки в коде длинной n
            file_bitarray[i: i+self.n] = code
            syndrome = syndromes.get(code.to01())     # Возвращение синдрома по ключу-коду
            if syndrome is not None:
                errors[i] = syndrome    # Если синдром найден, то заносим значение синдрома и ключ-индекс
            else:                       # иначе продолжаем
                continue
        # ["path", "txt"]
        path_to_file_arr = path_to_file.split(".")      # Разбиваем имя файла на название и формат файла
        with open(path_to_file_arr[0] + "_decode." + path_to_file_arr[1], 'wb') as f:
            file_bitarray.tofile(f)
        return errors


cc = CyclicCodes()
with open("Hello.txt", 'wb') as file:
    file_bitarray = ba.bitarray('0'*cc.n*225)   # создание bitarray с определённым количеством нулей
    for i in range(0, len(file_bitarray)):
        file_bitarray[i] = np.random.randint(0, 2) == 1     # Заполнение bitarray рандомными битами
    file_bitarray.tofile(file)                  # Запись псевдослучайный bitarray в файл
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
