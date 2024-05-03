import numpy as np
import pywt

def splitFreqBands(img, levRows, levCols):
    halfRow = int(levRows / 2)
    halfCol = int(levCols / 2)
    LL = img[0:halfRow, 0:halfCol]
    LH = img[0:halfRow, halfCol:levCols]
    HL = img[halfRow:levRows, 0:halfCol]
    HH = img[halfRow:levRows, halfCol:levCols]
    return LL, LH, HL, HH

def haarDWT1D(data, temp):
    avg0 = 0.5
    avg1 = 0.5
    dif0 = 0.5
    dif1 = -0.5
    h = len(data) // 2
    for i in range(h):
        k = i * 2
        temp[i] = data[k] * avg0 + data[k + 1] * avg1
        temp[i + h] = data[k] * dif0 + data[k + 1] * dif1
    data[:] = temp

def fwdHaarDWT2D(img):
    img = np.array(img, dtype=float)  # Assegura que a imagem esteja em float
    levRows, levCols = img.shape
    temp_row = np.empty(levCols, dtype=float)
    temp_col = np.empty(levRows, dtype=float)

    # Processamento das linhas
    for i in range(levRows):
        haarDWT1D(img[i, :], temp_row)
        img[i, :] = temp_row.copy()  # Atualiza a linha na imagem

    # Processamento das colunas
    for j in range(levCols):
        haarDWT1D(img[:, j], temp_col)
        img[:, j] = temp_col.copy()  # Atualiza a coluna na imagem

    return splitFreqBands(img, levRows, levCols)

# Exemplo de uso
# img = np.random.rand(256, 256)  # Substitua por sua imagem de entrada
# bands = fwdHaarDWT2D(img)
# LL, LH, HL, HH = bands
