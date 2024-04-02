import shutil
from ProjetoRedesNeurais.LabGPSI.auxiliary.globalVariables import current_path


def openTXT(path, filename):
    file_path = f'{path}/exif_txt/{filename}.txt'
    # file_path = f'{path}/{filename}.txt'
    file = open(file_path, 'r')

    return file


def openImageFile(path, filename):
    return open(path + '/' + filename, 'rb')


def writeCSV(content, modo, filename, path):

    try:
        file = open(path+'/'+filename, modo)
        if len(content) > 0:
            if type(content) is list:
                file.write(str(content[0]))
                for c in content[1:]:
                    file.write(f';{str(c)}')
                file.write('\n')
        file.close()
    except Exception as e:
        print('Error in writing csv: ', e)
    return


def writeTXTfile(filename, content, modo):
    file = open('{}.txt'.format(filename), modo)
    file.write('{}'.format(content))
    file.close()
    print(' {}.txt saved'.format(filename))
    return


def moveFile(oldPath=current_path, newPath='{}/Files'.format(current_path)):
    shutil.move(oldPath, newPath)
    # os.rename(oldPath, newPath)
    return
