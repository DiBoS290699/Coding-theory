import bitarray as ba
import numpy as np
from itertools import combinations


class RMCode:

    def __init__(self, r, m, major=False):
        if r < 0 or m < 0 or m < r:
            raise ValueError(f"The incorrect values r and m: r = {r}, m = {m}")
        self.r = r
        self.m = m
        self.k = 0
        for i in range(r + 1):
            self.k += np.math.factorial(m)/(np.math.factorial(i)*np.math.factorial(m - i))
        self.k = int(self.k)
        self.n = 2**m
        self.d = 2**(self.m - self.r)
        self.K_m = self.get_binary_digits(self.n)
        self.Z_m = np.arange(0, self.m)     # Массив чисел от 0 до m-1
        self.J_arr = self.J_array()     # Массив из J, где J включено в Z_m и |J| <= r
        self.major = major  # major == True - вызывается конструктор матрицы G канонического вида, иначе через рекурсию
        self.G_r_m = self.canon_G(self.J_arr, self.K_m) if major else self.recur_G(self.r, self.m)

    """Получение двоичного представления передаваемого числа с младшими разрядами слева"""
    def get_reverse_binary_number(self, j):
        str_j = bin(j)[2:]  # Приведение числа к булевому значению (но хранится как строка) с пропуском символов формата
        list = np.zeros(self.m, dtype=bool)
        for i in range(len(str_j)):     # Получение реверсивного вида битового числа (младшие разряды слева)
            list[i] = int(str_j[-i - 1]) == 1
        return list

    """Получение списка двоичных представлений для разрядов от 0 до n = 2**m"""
    def get_binary_digits(self, n):
        list = []
        for i in range(0, n):
            list.append(self.get_reverse_binary_number(i))
        return np.array(list)

    """Получение списка подмножеств J, где J включено в Z_m и |J| <= r"""
    def J_array(self):
        J_array = [()]
        if self.r > 0:
            for j in range(self.m - 1, -1, -1):
                J_array.append(tuple([j]))
        for i in range(2, self.r + 1):
            tmp_list = []
            for j in combinations(self.Z_m, i):
                tmp_list.append(j)
            tmp_list = self.quicksort_for_J_array(tmp_list)[::-1]
            J_array.extend(tmp_list)
        return np.array(J_array)

    """
    Сравнение двух подмножеств J1 и J2
    Возвращаемые значения:
        -1 если J1 < J2
        0 если J1 == J2
        1 если J1 > J2
    """
    def compare_J(self, J1, J2):
        if len(J1) == len(J2):
            # Так как подмножества J пресдтавленны в виде упорядоченного списка, обход начинается со старших элементов
            for i in range(len(J1) - 1, -1, -1):
                if J1[i] < J2[i]:       # Если наибольший элемент J2 больше наиб. элемента J1, то J1 < J2
                    return -1
                elif J1[i] > J2[i]:     # Если наиб. элемент J2 меньше наиб. элемента J1, то J1 > J2
                    return 1            # Если наиб. элементы равны, то переходим к следующему наиб. элементу
            return 0                    # Если не был произведён return до этого момента, то J1 == J2
        elif len(J1) > len(J2):
            return -1   # Подмножество с меньшей размерностью, чем у другого, главнее
        else:
            return 1    # Подмножество с меньшей размерностью, чем у другого, главнее

    """Быстрая сортировка массива J"""
    def quicksort_for_J_array(self, nums):
        if len(nums) <= 1:
            return nums
        else:
            q = nums[np.random.randint(0, len(nums))]
        l_nums = [n for n in nums if self.compare_J(n, q) == -1]

        e_nums = [q]
        b_nums = [n for n in nums if self.compare_J(n, q) == 1]
        return self.quicksort_for_J_array(l_nums) + e_nums + self.quicksort_for_J_array(b_nums)

    """
    Получение строки матрицы G каноничным способом
    Алгоритм:
        Значение элемента матрицы равно 1, если в двоичном представлении
        соотвествующего столбца все разряды, входящие в подмножество
        соответствующего столбца, равны нулю.
    """
    def row_canon_G(self, J, bin_digits):
        row = np.ones(len(bin_digits), dtype=bool)  # Изначально строка будет состоять только из True
        for j in range(len(bin_digits)):
            for k in J:
                row[j] &= not bin_digits[j, k]
        return row

    """Получение матрицы G каноничным способом"""
    def canon_G(self, J_array, bin_digits):
        G_r_m = []
        for i in range(len(J_array)):
            row = self.row_canon_G(J_array[i], bin_digits)
            G_r_m.append(row)
        return np.array(G_r_m, dtype=bool)

    """
    Получение десятичного числа из двоичного представления
    ( Выполняет обратное действие методу get_reverse_binary_number )
    Передаваемые значения:
        bits - любой массив битов
        reverse - означает с какой стороны находятся младшие разряды (True - слева, False - справа)
    """
    def get_number_from_binary(self, bits, reverse=False):
        number = 0
        indexes_of_bits = range(len(bits)) if not reverse else range(len(bits) - 1, -1, -1)
        degree = 0
        for i in indexes_of_bits:   # Стандартное преобразование двоичного числа в десятичное число
            if bits[i]:
                number += 2**degree
            degree += 1
        return number

    """
    Получение всех допустимых двоичных представлений
    сдвигов (в лекциях t), присвоив нули разрядам, входящим в множество J.
    """
    def get_binary_shifts(self, J):
        t = self.K_m.copy()
        if len(J) == 0:
            return t
        else:
            for elem_t in t:
                for elem_J in J:
                    elem_t[elem_J] = False
            return np.unique(t, axis=0)  # Удаление копий

    """
    Получение проверочные векторы (в лекциях v(Jc,t)) из 
    базового посредством сдвига влево (интерпретируя элементы из 
    множества t как двоичные представления сдвигов)
    Возвращаемое значение:
        Словарь пров. векторов, где ключ - сдвиг, значение - вектор со сдвигом
    """
    def verification_vectors_for_J(self, J):
        J_c = np.setdiff1d(self.Z_m, J)     # Получение комплементарного подмножества J_c = Z_m \ J
        b = self.row_canon_G(J_c, self.K_m)     # Базовый вектор, соответсвующий J_c
        t = self.get_binary_shifts(J)       # Получение двоичных представлений сдвигов
        ver_vectors = {}        # Словарь пров. векторов, где ключ - сдвиг, значение - вектор со сдвигом
        for elem_t in t:
            # np.roll() - операция сдвига элементов списка b на shift элементов
            ver_vectors[str(elem_t)] = np.roll(b, shift=self.get_number_from_binary(elem_t))
        return ver_vectors

    """Скалярное произведение двоичных векторов со сложением по модулю 2 (XOR)"""
    def dot_with_mod2(self, a, b):
        if len(a) != len(b):
            raise Exception("The dimensions of the transmitted values do not match")
        else:
            result = False
            for i in range(len(a)):
                result ^= a[i] & b[i]
            return result

    """Выполнение алгоритма мажоритарного декодирования"""
    def major_decode(self, message):
        i = self.r      # Текущая размерность подмножеств J
        w_i = np.array(message)     # Битовая последовательность, которая будет изменяться, согласно значению i
        m = np.zeros(self.k, dtype=bool)    # Декодированное сообщение (изначально полностью забит нулями)
        sum_rows_with_1 = np.zeros(self.n, dtype=bool)      # Сумма всех m_J*v_J, где |J| = i
        for index_J in range(len(self.J_arr) - 1, -1, -1):
            J = self.J_arr[index_J]     # Текущее подмножество J
            if len(J) < i:      # Если текущий J, размерностью меньше i, то переходим к i-1 меняем w(i) на w(i-1)
                w_i = w_i ^ sum_rows_with_1
                sum_rows_with_1 = np.zeros(self.n, dtype=bool)
                if not np.sum(w_i) > 2**(self.m - self.r - 1) - 1:
                    # Если вес w(i) не больше 2**(m-r-1) - 1, то останавливаемся
                    return m
                else:
                    i -= 1
            ver_vectors = self.verification_vectors_for_J(J)
            dot_vv_with_w = []
            for vector in ver_vectors.values():
                dot_vv_with_w.append(self.dot_with_mod2(w_i, vector))
            count_True = dot_vv_with_w.count(True)
            count_False = len(dot_vv_with_w) - count_True
            if count_True > count_False:
                sum_rows_with_1 ^= self.row_canon_G(J, self.K_m)
                m[index_J] = True
            elif count_True < count_False:
                m[index_J] = False
            else:
                raise Exception("Send the message again")
        return m

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
            return np.concatenate([self.recur_G(m - 1, m), bottom_line]) # Конкатенация массивов, при axis=0 (По умол.) - снизу, при axis=1 - справа
        else:
            G_r_m_1 = np.concatenate([self.recur_G(r, m - 1), self.recur_G(r, m - 1)], axis=1) # Конкатенация массивов, при axis=0 (По умол.) - снизу, при axis=1 - справа
            G_r_1_m_1 = self.recur_G(r - 1, m - 1)
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

    def recur_decode(self, input_n):
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
        error_bits = []
        i = 0
        while i < count_errors:
            error_bit = np.random.randint(0, self.n)  # Индекс бита, в котором будет ошибка
            if error_bit not in error_bits:
                code[error_bit] = not code[error_bit]
                error_bits.append(error_bit)
                i += 1
        return error_bits, code

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
                _, encode = self.make_a_mistake(encode, count_errors)
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
            decode = self.major_decode(code) if self.major else self.recur_decode(code)
            decode_file.extend(decode)
        with open(path_to_output_file, 'wb') as f:
            decode_file = self.array2bitarray(decode_file)
            decode_file.tofile(f)
        print(f"Decoding ({path_to_input_file} in {path_to_output_file}) completed successfully")


r = 2
m = 4
count_errors = 1

rmc = None
try:
    rmc = RMCode(r, m, major=True)
except ValueError as e:
    print(e)
else:
    print(f"The canonical matrix G({r},{m})\n {rmc.G_r_m.astype(int)}")

    message = np.zeros(rmc.k, dtype=bool)
    for i in range(len(message)):
        message[i] = np.random.randint(0, 2, dtype=bool)
    print(f"Message:\n {message.astype(int)}")

    encode = rmc.encode(message)
    print(f"encode:\n {encode.astype(int)}")

    indexes_error, encode = rmc.make_a_mistake(encode, count_errors=count_errors)
    print(f"encode with error in {indexes_error} bit:\n {encode.astype(int)}")

    try:
        decode = rmc.major_decode(encode)
        print(f"decode:\n {decode.astype(int)}")
        print(f"message == decode:\n {np.equal(message, decode)}")
    except Exception:
        print("ERROR! Send the message again")
