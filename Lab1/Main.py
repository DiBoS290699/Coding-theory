import bitarray as ba
import numpy as np

bits = ba.bitarray()
f = open("test.txt", 'rb')
bits.fromfile(f)
print(bits)



def from_H_in_G():
    matrix_H = [[0, 1, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 1]]

    matrix_G = [[1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]]

    for i in range(4):
        row_H = 0
        for k in range(4, 7):
            matrix_G[i][k] = matrix_H[row_H][i]
            row_H += 1

    return matrix_H, matrix_G


# def from_H_in_G_with_b():
#     matrix_H = ba.bitarray(['001', '010', '011', '100', '101', '110', '111'])
#     matrix_G = ba.bitarray(['1000', '0100', '0010', '0001', '', '', ''])
#
#     return matrix_G


def coding(bitarray):
    new_bitarray = ba.bitarray()
    matrix_H, matrix_G = from_H_in_G()
    for i in range(4, len(bitarray) + 1, 4):
        vector = []
        for g in range(i - 4, i):
            a = int(bitarray[g])
            vector.append(a)
        c = np.dot(vector, matrix_G) % 2
        new_bitarray.extend(c)

    return new_bitarray

def decoding(file_name):
    file_bitarray = ba.bitarray()
    with open(file_name, 'rb') as file:
        file_bitarray.fromfile(file)
    decod_bitarray = ba.bitarray()
    for i in range(0, len(file_bitarray), 7):
        decod_bitarray.extend(file_bitarray[i: i + 4])

    with open(file_name + "_decoding", 'wb') as file:
        decod_bitarray.tofile(file)
    return decod_bitarray


def add_rand_error(file_name):
    file_bitarray = ba.bitarray()
    with open(file_name, 'rb') as file:
        file_bitarray.fromfile(file)
    print(f'Source file: {file_bitarray}')
    index_error = np.random.randint(0, len(file_bitarray))
    file_bitarray[index_error] = 0 if file_bitarray[index_error] == 1 else 1
    print(f'New file with the error index {index_error}: {file_bitarray}')
    with open(file_name, 'wb') as file:
        file_bitarray.tofile(file)


def search_error(file_name, matrix_H):
    file_bitarray = ba.bitarray()
    with open(file_name, 'rb') as file:
        file_bitarray.fromfile(file)
    syndromes = {}
    for i in range(0, len(file_bitarray), 7):
        vector_c = []
        for k in range(i, i + 7):
            vector_c.append(int(file_bitarray[k]))
        syndrome = np.dot(matrix_H, np.transpose(vector_c)) % 2
        bit_of_syndrome = 
        syndromes.append


matrix_H, matrix_G = from_H_in_G()
# print(matrix_G)
# a = [0, 1, 1, 0]
# c = np.dot(a, matrix_G) % 2
# print(c)
# print(f'np.dot(matrix_H, np.transpose(c)) % 2 == {np.dot(matrix_H, np.transpose(c)) % 2}')
# err_index = np.random.randint(0, len(c))
# c[err_index] = 0 if c[err_index] == 1 else 1
# print(f'Индекс ошибки: {err_index}')
# print(f'C с ошибкой в индексе {err_index}: {c}')
# print(f'np.dot(matrix_H, np.transpose(err_c)) % 2 == {np.dot(matrix_H, np.transpose(c)) % 2}')
new_bitarray = coding(bits)
file_name = 'Test.zip'
with open(file_name, 'wb') as test_write:
    new_bitarray.tofile(test_write)
print(new_bitarray)
decod_bitarray = decoding(file_name)
print(decod_bitarray)

add_rand_error(file_name)
search_error(file_name, matrix_H)