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

    # Возвращение кода со случайной ошибкой, либо без неё
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

    # Кодировка файла, обнаружение ошибки, декодировка файла
    # Взять файл, 4 бита систематически кодировать в 7, записать файл, прочитать этот файл,
    # каждые 7 битов поделит и получить остаток, по остатку (синдрому) найти вектор в таблице синдромов.
    # Если синдром нулевой, то ошибки не было, иначе прибавляем к 7 битам полученный вектор ошибки,
    # который хранит 1 в месте ошибки. Затем берём последние 4 бита из 7 битов и записывыаем в файл раскодированный код
    # def search_error(self, path_to_file):
    #     noiseproof_bitarray = ba.bitarray()
    #     error_bitarray = None
    #     syndromes = self.make_table()   # Возвращение таблицы синдромов класса
    #     errors = {}     # Словарь ошибок: key - индекс начала кода с ошибкой, значение - синдромкода с ошибкой
    #     with open(path_to_file, 'rb') as file:
    #         noiseproof_bitarray.fromfile(file)        # Чтение файла
    #     error_bitarray = ba.bitarray('0'*int((len(noiseproof_bitarray) * 7 / 4)))
    #     for i in range(0, len(noiseproof_bitarray), self.k):
    #         code = noiseproof_bitarray[i: i+self.k]
    #         if len(code) != self.k:
    #             continue
    #         encode = self.encode_sys(code)
    #         error_encode = self.make_a_mistake(encode)    # Создание ошибки в коде длинной n
    #         noiseproof_bitarray[i: i+self.k] = encode
    #         error_bitarray[i: i+self.n] = error_encode
    #         syndrome = self.remainder(error_encode)
    #         syndrome = syndromes.get(code.to01())     # Возвращение синдрома по ключу-коду
    #         if syndrome is not None:
    #             errors[i] = syndrome    # Если синдром найден, то заносим значение синдрома и ключ-индекс
    #         else:                       # иначе продолжаем
    #             continue
    #     # ["path", "txt"]
    #     path_to_file_arr = path_to_file.split(".")      # Разбиваем имя файла на название и формат файла
    #     with open(path_to_file_arr[0] + "_decode." + path_to_file_arr[1], 'wb') as f:
    #         noiseproof_bitarray.tofile(f)
    #     return errors

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
