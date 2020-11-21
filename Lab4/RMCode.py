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
        # Построение порождающей матрицы рекурсивным методом
        if m is None or r is None or r < 0 or m < 0 or m < r or m == r == 0:
            print(f"ERROR! The value m or r is incorrect: m is {m}, r is {r}")
            return None
        elif r == 0:
            line = np.ones((1, 2**m), dtype=bool)   # Возвращает массив единиц с указанным размером
            return line
        elif r == m:
            bottom_line = np.zeros((1, 2**m), dtype=bool)   # Возвращает массив нулей с указанным размером
            bottom_line[0][-1] = True       # Последний бит равен True
            return np.concatenate([self.gen_matrix(m - 1, m), bottom_line]) # Конкатенация массивов, при axis=0 (По умол.) - снизу, при axis=1 - справа
        else:
            G_r_m_1 = np.concatenate([self.gen_matrix(r, m - 1), self.gen_matrix(r, m - 1)], axis=1) # Конкатенация массивов, при axis=0 (По умол.) - снизу, при axis=1 - справа
            G_r_1_m_1 = self.gen_matrix(r - 1, m - 1)
            shape_zeros = np.shape(G_r_1_m_1)       # Получение размерности матрицы G
            zeros = np.zeros(shape_zeros, dtype=bool)
            return np.concatenate([G_r_m_1, np.concatenate([zeros, G_r_1_m_1], axis=1)])

    def encode(self, input_k):
        # Кодирование, путём умножения на порождающую матрицу
        if self.k != len(input_k):
            print(f'ERROR! The input_k has an incorrect size. The size must be equal to {self.k} instead of {len(input_k)}')
            return None
        if input_k.__class__ == ba.bitarray().__class__:  # Если передаваемый параметр - bitarray, то вызываем tolist()
            np_input_k = np.array(input_k.tolist(), dtype=int)
        else:
            np_input_k = np.array(input_k, dtype=int)
        np_G_r_m = np.array(self.G_r_m, dtype=int)
        dot = np.dot(np_input_k, np_G_r_m) % 2      # Умножение матриц и приведение к булевым значениям
        return np.array(dot, dtype=bool)

    def H_i_m(self, i, m):
        # Получение H^i_m с помощью умножения Кронекера
        if i < 1 or m < i:
            print(f"ERROR! The value i must be 1 <= i <= m. i is {i}, m is {m}")
            return None
        else:
            I_left = np.eye(2**(m - i))     # Единичная матрица определённого размера
            H = np.array([[1, 1], [1, -1]])
            I_right = np.eye(2**(i - 1))    # Единичная матрица определённого размера
            return np.kron(np.kron(I_left, H), I_right)  # Последовательное произведение Кронекера

    def v_j(self, j):
        # Двоичное представление передаваемого числа с младшими разрядами слева
        str_j = bin(j)[2:]  # Приведение числа к булевому значению (но хранится как строка) с пропуском символов формата
        list = np.zeros(self.m, dtype=bool)
        for i in range(len(str_j)):     # Получение реверсивного вида битового числа (младшие разряды слева)
            list[i] = int(str_j[-i - 1]) == 1
        return list

    def decode(self, input_n):
        # Алгоритм декодирования
        if len(input_n) != self.n:
            print(f"ERROR! The input_n has an incorrect size. The size must be equal to {self.n} instead of {len(input_n)}")
            return None
        new_input_n = np.ones(self.n, dtype=int)
        for i in range(self.n):     # Получение нового кода, в котором 0 заменяются на -1
            new_input_n[i] = 1 if input_n[i] == 1 else -1

        w_i = np.dot(new_input_n, self.H_i_m(1, self.m))    # Получение w_1 путём умножение нового кода на H^1_m
        for i in range(2, self.m + 1):
            w_i = np.dot(w_i, self.H_i_m(i, self.m))        # Получение w_i путём умножение w_(i-1) на H^i_m
        j = np.argmax(np.abs(w_i))      # Получение индекса максимального абсолютного значения в массиве
        v_j = self.v_j(j)       # Получение реверсивного битового значения индекса j

        result = np.zeros(len(v_j) + 1, dtype=bool)
        for i in range(1, len(v_j) + 1):        # Получение новго массива с True в начале при w_i[j] > 0, иначе False
            result[i] = v_j[i - 1]
        result[0] = w_i[j] > 0
        return result

    def make_a_mistake(self, code, count_errors=1):
        # Возвращение кода со случайным количеством ошибок (максимум count_errors ошибок)
        if len(code) != self.n:
            print("ERROR! Invalid code length.")
            return code
        for i in range(count_errors):
            error_bit = np.random.randint(0, self.n)  # Индекс бита, в котором будет ошибка
            code[error_bit] = not code[error_bit]
        return code

    def array2bitarray(self, array):
        # Преобразование массива в bitarray
        ba_list = ba.bitarray('0'*len(array))
        for k in range(len(array)):
            ba_list[k] = array[k]
        return ba_list

    def encode_file(self, path_to_input_file, path_to_output_file, error=False, count_errors=1):
        # Кодировка файла из path_to_input_file в path_to_output_file с допуском ошибки (error is True) или без ошибок
        file_bitarray = ba.bitarray()
        with open(path_to_input_file, 'rb') as file:
            file_bitarray.fromfile(file)  # Чтение файла
        if not error:
            for i in range(0, len(file_bitarray), self.n):
                code = file_bitarray[i: i + self.k]
                if len(code) != self.k:
                    continue
                encode = self.encode(code)
                file_bitarray[i:i + self.k] = self.array2bitarray(encode)
        else:
            for i in range(0, len(file_bitarray), self.n):
                code = file_bitarray[i: i + self.k]
                if len(code) != self.k:
                    continue
                encode = self.encode(code)
                encode = self.make_a_mistake(encode, count_errors)
                file_bitarray[i:i + self.k] = self.array2bitarray(encode)
        with open(path_to_output_file, 'wb') as f:
            file_bitarray.tofile(f)
        print(f"Encoding ({path_to_input_file} in {path_to_output_file}) completed successfully")

    def decode_file(self, path_to_input_file, path_to_output_file):
        # Декодирование файла path_to_input_file в файл path_to_output_file
        file_bitarray = ba.bitarray()
        with open(path_to_input_file, 'rb') as file:
            file_bitarray.fromfile(file)  # Чтение файла
        for i in range(0, len(file_bitarray), self.k):
            code = file_bitarray[i: i + self.n]
            if len(code) != self.n:
                continue
            decode = self.decode(code)
            file_bitarray[i:i + self.n] = self.array2bitarray(decode)
        with open(path_to_output_file, 'wb') as f:
            file_bitarray.tofile(f)
        print(f"Decoding ({path_to_input_file} in {path_to_output_file}) completed successfully")


rmc = RMCode(1, 3)
# G_r_m = rmc.gen_matrix(3, 3)
# print(G_r_m)
input_k = []
for i in range(rmc.k):
    input_k.append(np.random.randint(0, 2, dtype=bool))
ba_input_k = ba.bitarray(input_k)
print(f"The input_k == \n{ba_input_k}")

encode = rmc.encode(ba_input_k)
print(f"The encode == \n{np.array(encode, dtype=int)}")

# print(f'The encode: {10001111}')
# decode = rmc.decode(ba.bitarray('10001111'))

one_error = [np.random.randint(0, len(encode))]
two_errors = [0, 3]
three_errors = [1, 2, 4]
for_errors = [0, 5, 6, 7]

tmp = rmc.encode(ba_input_k)
for i in one_error:
    tmp[i] = not tmp[i]
print(f'The encode with an error in the {one_error} bits: \n{np.array(tmp, dtype=int)}')
decode = rmc.decode(tmp)
print(f'The decode without errors: \n{np.array(decode, dtype=int)}')

tmp = rmc.encode(ba_input_k)
for i in two_errors:
    tmp[i] = not tmp[i]
print(f'The encode with an error in the {two_errors} bits: \n{np.array(tmp, dtype=int)}')
decode = rmc.decode(tmp)
print(f'The decode without errors: \n{np.array(decode, dtype=int)}')

tmp = rmc.encode(ba_input_k)
for i in three_errors:
    tmp[i] = not tmp[i]
print(f'The encode with an error in the {three_errors} bits: \n{np.array(tmp, dtype=int)}')
decode = rmc.decode(tmp)
print(f'The decode without errors: \n{np.array(decode, dtype=int)}')

tmp = rmc.encode(ba_input_k)
for i in for_errors:
    tmp[i] = not tmp[i]
print(f'The encode with an error in the {for_errors} bits: \n{np.array(tmp, dtype=int)}')
decode = rmc.decode(tmp)
print(f'The decode without errors: \n{np.array(decode, dtype=int)}')


path_to_input_file = "Hello.txt"
path_to_output_file = "Hello_encode.txt"
path_to_output_file_with_errors = "Hello_errors.txt"
path_to_output_file_without_errors = "Hello_without_errors.txt"

# with open(path_to_input_file, 'wb') as file:
#     file_bitarray = ba.bitarray('0'*rmc.k*3)   # создание bitarray с определённым количеством нулей
#     for i in range(0, len(file_bitarray)):
#         file_bitarray[i] = np.random.randint(0, 2) == 1     # Заполнение bitarray рандомными битами
#     file_bitarray.tofile(file)                  # Запись псевдослучайный bitarray в файл

rmc.encode_file(path_to_input_file, path_to_output_file, error=False)
rmc.encode_file(path_to_input_file, path_to_output_file_with_errors, error=True)
rmc.decode_file(path_to_output_file, path_to_output_file_without_errors)
