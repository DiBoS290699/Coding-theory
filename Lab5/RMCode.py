import bitarray as ba
import numpy as np
from itertools import combinations


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
        self.K_m = self.get_binary_digits()
        self.Z_m = np.arange(0, self.m)
        self.J_arr = self.J_array()
        self.G_r_m = self.canon_G(self.J_arr, self.K_m)


    def recur_G(self, r, m):
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

    def get_binary_digits(self):
        list = []
        for i in range(0, self.n):
            list.append(self.get_reverse_binary_number(i))
        return np.array(list)

    def J_array(self):
        J_array = []
        for i in range(self.r + 1):
            tmp_list = []
            for j in combinations(self.Z_m, i):
                tmp_list.append(j)
            tmp_list = tmp_list[::-1]
            J_array.extend(tmp_list)
        return np.array(J_array)

    def row_canon_G(self, J, bin_digits):
        row = np.ones(len(bin_digits), dtype=bool)
        for j in range(len(bin_digits)):
            for k in J:
                if k is None:
                    row[j] &= True
                else:
                    row[j] &= not bin_digits[j, k]
        return row

    def canon_G(self, J_array, bin_digits):
        G_r_m = []
        for i in range(len(J_array)):
            row = self.row_canon_G(J_array[i], bin_digits)
            G_r_m.append(row)
        return G_r_m

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

    def get_reverse_binary_number(self, j):
        # Двоичное представление передаваемого числа с младшими разрядами слева
        str_j = bin(j)[2:]  # Приведение числа к булевому значению (но хранится как строка) с пропуском символов формата
        list = np.zeros(self.m, dtype=bool)
        for i in range(len(str_j)):     # Получение реверсивного вида битового числа (младшие разряды слева)
            list[i] = int(str_j[-i - 1]) == 1
        return list

    def get_number_from_binary(self, bits, reverse=False):
        number = 0
        indexes_of_bits = range(len(bits)) if not reverse else range(len(bits) - 1, -1, -1)
        degree = 0
        for i in indexes_of_bits:
            if bits[i]:
                number += 2**degree
            degree += 1
        return number

    def get_binary_shifts(self, J):
        t = self.K_m.copy()
        if len(J) == 0:
            return t
        else:
            for elem_t in t:
                for elem_J in J:
                    elem_t[elem_J] = False
            return np.unique(t, axis=0)

    def verification_vectors_for_J(self, J):
        J_c = np.setdiff1d(self.Z_m, J)
        b = self.row_canon_G(J_c, self.K_m)
        t = self.get_binary_shifts(J)
        ver_vectors = {}
        for elem_t in t:
            ver_vectors[str(elem_t)] = np.roll(b, shift=self.get_number_from_binary(elem_t))
        return ver_vectors

    def dot_with_mod2(self, a, b):
        if len(a) != len(b):
            raise Exception("The dimensions of the transmitted values do not match")
        else:
            result = False
            for i in range(len(a)):
                result ^= a[i] & b[i]
            return result

    def major_decode(self, message):
        i = self.r
        w_i = np.array(message)
        m = np.zeros(self.n, dtype=bool)
        sum_rows_with_1 = np.zeros(self.n, dtype=bool)
        for index_J in range(len(self.J_arr) - 1, -1, -1):
            J = self.J_arr[index_J]
            if len(J) < i or J[-1] is None:
                w_i = w_i + sum_rows_with_1
                if not np.sum(w_i) > 2**(self.m - self.r - 1) - 1:
                    print("w(i-1) has a weight of no more than 2**(m - r - 1) - 1)")
                    return m
                else:
                    i -= 1
                    index_J += 1
                    continue
            else:
                ver_vectors = self.verification_vectors_for_J(J)
                dot_vv_with_w = []
                for vector in ver_vectors.values():
                    dot_vv_with_w.append(self.dot_with_mod2(w_i, vector))
                count_True = dot_vv_with_w.count(True)
                count_False = len(dot_vv_with_w) - count_True
                if count_True > count_False:
                    sum_rows_with_1 ^= self.verification_vectors_for_J(np.setdiff1d(self.Z_m, J))["[False False False False]"]
                    m[index_J] = True
                elif count_True < count_False:
                    m[index_J] = False
                else:
                    raise Exception("Send the message again")
        return m



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
        v_j = self.get_reverse_binary_number(j)       # Получение реверсивного битового значения индекса j

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
        size = len(file_bitarray)
        encode_file = []
        if not error:
            for i in range(0, size, self.k):
                code = file_bitarray[i: i + self.k]
                if len(code) != self.k:
                    continue
                encode = self.encode(code)
                encode_file.extend(encode)
        else:
            for i in range(0, size, self.k):
                code = file_bitarray[i: i + self.k]
                if len(code) != self.k:
                    continue
                encode = self.encode(code)
                encode = self.make_a_mistake(encode, count_errors)
                encode_file.extend(encode)
        with open(path_to_output_file, 'wb') as f:
            encode_file = self.array2bitarray(encode_file)
            encode_file.tofile(f)
        print(f"Encoding ({path_to_input_file} in {path_to_output_file}) completed successfully")

    def decode_file(self, path_to_input_file, path_to_output_file):
        # Декодирование файла path_to_input_file в файл path_to_output_file
        file_bitarray = ba.bitarray()
        with open(path_to_input_file, 'rb') as file:
            file_bitarray.fromfile(file)  # Чтение файла
        size = len(file_bitarray)
        decode_file = []
        for i in range(0, size, self.n):
            code = file_bitarray[i: i + self.n]
            if len(code) != self.n:
                continue
            decode = self.decode(code)
            decode_file.extend(decode)
        with open(path_to_output_file, 'wb') as f:
            decode_file = self.array2bitarray(decode_file)
            decode_file.tofile(f)
        print(f"Decoding ({path_to_input_file} in {path_to_output_file}) completed successfully")


rmc = RMCode(2, 4)
message1 = [False, True, False, True, False, True, True, True, True, False, True, False,
           False, False, False, False]
message2 = [True, False, True, False, True, False, False, False, False, True, False,
            True, True, True, True, True, ]
decode = rmc.major_decode(message2)
print(np.array(decode, dtype=int))
# print(np.array(rmc.G_r_m, dtype=int))
# G_r_m = rmc.gen_matrix(3, 3)
# print(G_r_m)
# input_k = []
# for i in range(rmc.k):
#     input_k.append(np.random.randint(0, 2, dtype=bool))
# ba_input_k = ba.bitarray(input_k)
# print(f"The input_k == \n{ba_input_k}")
#
# encode = rmc.encode(ba_input_k)
# print(f"The encode == \n{np.array(encode, dtype=int)}")
#
# # print(f'The encode: {10001111}')
# # decode = rmc.decode(ba.bitarray('10001111'))
#
# one_error = [np.random.randint(0, len(encode))]
# two_errors = [0, 3]
# three_errors = [1, 2, 4]
# for_errors = [0, 5, 6, 7]
#
# tmp = rmc.encode(ba_input_k)
# for i in one_error:
#     tmp[i] = not tmp[i]
# print(f'The encode with an error in the {one_error} bits: \n{np.array(tmp, dtype=int)}')
# decode = rmc.decode(tmp)
# print(f'The decode without errors: \n{np.array(decode, dtype=int)}')
#
# tmp = rmc.encode(ba_input_k)
# for i in two_errors:
#     tmp[i] = not tmp[i]
# print(f'The encode with an error in the {two_errors} bits: \n{np.array(tmp, dtype=int)}')
# decode = rmc.decode(tmp)
# print(f'The decode without errors: \n{np.array(decode, dtype=int)}')
#
# tmp = rmc.encode(ba_input_k)
# for i in three_errors:
#     tmp[i] = not tmp[i]
# print(f'The encode with an error in the {three_errors} bits: \n{np.array(tmp, dtype=int)}')
# decode = rmc.decode(tmp)
# print(f'The decode without errors: \n{np.array(decode, dtype=int)}')
#
# tmp = rmc.encode(ba_input_k)
# for i in for_errors:
#     tmp[i] = not tmp[i]
# print(f'The encode with an error in the {for_errors} bits: \n{np.array(tmp, dtype=int)}')
# decode = rmc.decode(tmp)
# print(f'The decode without errors: \n{np.array(decode, dtype=int)}')
#
#
# path_to_input_file = "Hello.txt"
# path_to_output_file = "Hello_encode.txt"
# path_to_output_file_with_errors = "Hello_errors.txt"
# path_to_output_file_without_errors = "Hello_without_errors.txt"
#
# # with open(path_to_input_file, 'wb') as file:
# #     file_bitarray = ba.bitarray('0'*rmc.k*3)   # создание bitarray с определённым количеством нулей
# #     for i in range(0, len(file_bitarray)):
# #         file_bitarray[i] = np.random.randint(0, 2) == 1     # Заполнение bitarray рандомными битами
# #     file_bitarray.tofile(file)                  # Запись псевдослучайный bitarray в файл
#
# rmc.encode_file(path_to_input_file, path_to_output_file, error=False)
# rmc.encode_file(path_to_input_file, path_to_output_file_with_errors, error=True)
# rmc.decode_file(path_to_output_file_with_errors, path_to_output_file_without_errors)
