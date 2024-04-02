import os

from ProjetoRedesNeurais.T1.auxiliary.globalVariables import base_path, current_path
from ProjetoRedesNeurais.T1.auxiliary.globalVariables import csv_filename1, csv_filename2
from ProjetoRedesNeurais.T1.auxiliary.globalVariables import alphabet

from ProjetoRedesNeurais.auxiliary_func.getPath import editPath

image_path = f'{editPath(current_path)}/Images'.replace('\\', '/')
filenames = [name for name in os.listdir(image_path) if os.path.splitext(name)[-1] == '.jpg']
exif_path = f'{image_path}/exif_txt'


# Function to get indexes according to input given
def getSpecificIndexes(input, header):
    # file = openTXT(image_path, filenames[0])  # gets header from the first exif file
    # header = createContentList(file, 0)

    indexes = []
    for i, label in enumerate(header):
        if input in label:
            indexes.append(i)
    return indexes


def getSpecificIndexesFromContent(input, content):
    indexes = []
    for i, data in enumerate(content):
        if input in data:
            indexes.append(i)
    return indexes


def getMultSpecificIndexesFromContent(lInput, content):
    indexes = []
    for input in lInput:
        aux = getSpecificIndexesFromContent(input, content)
        for index in aux:
            if index not in indexes:
                indexes.append(index)
    return indexes


def getPartiallyNumericIndexesFromContent(content, count1, count2, str_input='/ :'):
    indexes = []

    for index, data in enumerate(content):
        # print('getPartiallyNumericIndexesFromContent:', index, data)
        if index == '/ :':
            if (index not in indexes) and (content.count('/') == count1 and content.count(':') == count2):
                indexes.append(index)
    return indexes


def getDateIndexesHeader(content):
    dateIndexes = sorted(getMultSpecificIndexesFromContent(['DateTime'], content))
    return tuple(dateIndexes)


def getDateIndexesContent(content):
    dateIndexes = sorted(getDateTimeIndexesFromContent(["/:"], content))
    return tuple(dateIndexes)


def getOffsetTimeIndexes(content):
    offsetTimeIndexes = sorted(getMultSpecificIndexesFromContent(['OffsetTime'], content))
    return tuple(offsetTimeIndexes)


def getDivDataIndexes(content):
    divDataIndexes = sorted(getMultSpecificIndexesFromContent(['ShutterSpeed', 'Brightness', 'ExposureBias'], content))
    return tuple(divDataIndexes)


def findIndexFilePath(header):
    FilePathIndex = getSpecificIndexes('FilePath', header)
    return FilePathIndex[0]


def findStrIndexes(content):
    strIndexes = sorted(getMultSpecificIndexesFromContent(alphabet, content))
    # dateIndexes = getDateIndexes(content)

    # for i in dateIndexes:
    #     if i in strIndexes:
    #         strIndexes.remove(i)

    return tuple(strIndexes)
