import bitarray as ba
import numpy as np


def dot_for_bits(input_1, input_2):
    shape_1 = np.shape(input_1)
    if len(shape_1) == 1:
        shape_1 = (1, shape_1[0])
    shape_2 = np.shape(input_2)
    if len(shape_2) == 1:
        shape_2 = (shape_1[0], 1)
    if shape_1[1] != shape_2[0]:
        print('ERROR!Incorrect dimensions')
        return None
    result = np.zeros((shape_1[0], shape_2[1]), dtype=bool)
    for i in range(shape_1[0]):
        if shape_1[0] == 1:
            for j in range(shape_2[1]):
                if shape_2[] ==
                for s in range(shape_1[1]):
                    tmp = input_1[s] & input_2[s][j]
                    result[j] ^= tmp
        for j in range(shape_2[1]):
            for s in range(shape_1[1]):
                tmp = input_1[i][s] & input_2[s][j]
                result[i][j] ^= tmp
    return result


B = (
        ba.bitarray('110111000101'),
        ba.bitarray('101110001011'),
        ba.bitarray('011100010111'),
        ba.bitarray('111000101101'),
        ba.bitarray('110001011011'),
        ba.bitarray('100010110111'),
        ba.bitarray('000101101111'),
        ba.bitarray('001011011101'),
        ba.bitarray('010110111001'),
        ba.bitarray('101101110001'),
        ba.bitarray('011011100011'),
        ba.bitarray('111111111110')
        )

G = np.zeros((12, 24), dtype=bool)
for i in range(0, 12):
    G[i][i] = True
    b_i = B[i].tolist()
    for j in range(12):
        G[i][12 + j] = b_i[j]

input_k = [False, True, False, False, True, False, True, True, True, False, False, True]
result = dot_for_bits(input_k, G)
print(result)
