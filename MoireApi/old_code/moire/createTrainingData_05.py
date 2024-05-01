import sys
import argparse
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from haar2D import fwdHaarDWT2D
import multiprocessing
import numpy as np
import datetime

def log(message, file='process_log.txt'):
    """ Escreve uma mensagem no arquivo de log com a data e hora atual. """
    with open(file, 'a') as f:
        f.write(f"{datetime.datetime.now()}: {message}\n")

def normalize_wavelet_components(components):
    return [(component - np.min(component)) / (np.max(component) - np.min(component)) for component in components]

def transformImageAndSave(image, filename, customStr, path):
    try:
        cA, cH, cV, cD = fwdHaarDWT2D(image)
        components = normalize_wavelet_components([cA, cH, cV, cD])
        suffixes = ['LL', 'LH', 'HL', 'HH']
        
        for suffix, component in zip(suffixes, components):
            f_new = f'{os.path.splitext(filename)[0]}_{customStr}{suffix}.tiff'
            Image.fromarray(np.uint8(component * 255)).save(join(path, f_new), format='TIFF')
        return True
    except Exception as e:
        log(f'Erro ao salvar imagem {filename}: {e}')
        return False

def augmentAndTransformImage(args):
    f, mainFolder, trainFolder = args
    try:
        img = Image.open(join(mainFolder, f))
        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        imgGray = img.convert('L')
        
        if imgGray.size[1] > imgGray.size[0]:
            imgGray = imgGray.rotate(-90, expand=True)
        
        success = transformImageAndSave(imgGray, f, '', trainFolder)
        if not success:
            raise ValueError("Falha ao transformar imagem base.")
        
        for transform, suffix in [(Image.ROTATE_180, '180_'), (Image.FLIP_LEFT_RIGHT, '180_FLIP_')]:
            transformed_img = imgGray.transpose(transform)
            success = transformImageAndSave(transformed_img, f, suffix, trainFolder)
            if not success:
                raise ValueError(f"Falha ao transformar imagem com transformação {suffix}.")
        return True
    except Exception as e:
        log(f'Erro ao processar arquivo {f}: {e}')
        return False

def processImageData(baseFolderPath, directory):
    log('Iniciando processamento de imagem.')
    folderPath = join(baseFolderPath, directory)
    for category in ['positive', 'negative']:
        imageFolder = join(folderPath, category)
        outputFolder = join('./processedData', directory, category)
        os.makedirs(outputFolder, exist_ok=True)
        imageFiles = [f for f in listdir(imageFolder) if isfile(join(imageFolder, f))]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(augmentAndTransformImage, [(f, imageFolder, outputFolder) for f in imageFiles])
        log(f'Processadas {sum(results)} imagens em {imageFolder}')
    log('Processamento de imagem concluído.')

def main(argv):
    parser = argparse.ArgumentParser(description='Processa imagens aplicando transformada de Haar e pré-processamento.')
    parser.add_argument('baseFolderPath', type=str, help='Diretório base contendo subpastas para treinamento, teste e validação.')
    parser.add_argument('directory', type=str, help='Diretório a ser processado (train, test ou validation).')
    args = parser.parse_args(argv)
    processImageData(args.baseFolderPath, args.directory)

if __name__ == '__main__':
    main(sys.argv[1:])
