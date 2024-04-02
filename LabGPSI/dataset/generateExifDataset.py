import os
from ProjetoRedesNeurais.LabGPSI.auxiliary.readFiles import writeCSV

from ProjetoRedesNeurais.LabGPSI.auxiliary.globalVariables import current_path, image_path
from ProjetoRedesNeurais.LabGPSI.auxiliary.globalVariables import csv_filename2

from ProjetoRedesNeurais.LabGPSI.exif.getExif import getExifDict


# Gets the list of labels in header
def get_header_list(filename, exif_data):
    try:
        exif_label = ['Filename'] + [label
                                     for label in exif_data.get(filename).keys()
                                     if exif_data.get(filename) is not None
                                     ] + ['Observations']
        return exif_label
    except Exception as e:
        print('Error in get_header_list: ', e)
        return


# Gets the list of exif's data of each image
def get_content_list(filename, exif_data):

    try:
        img_dict_exif = exif_data.get(filename)
        content_list = []

        if img_dict_exif:
            content_list = [filename] + list(img_dict_exif.values()) + ['sem obs']

        return content_list
    except Exception as e:
        print('Error on get_content_list: ', e)
        return


def generateExifDataset():

    try:
        exif_data = getExifDict()

        # Run this part only once to add header
        filenames = list(exif_data.keys())
        header = get_header_list(filenames[0], exif_data)

        # print(header)
        writeCSV(header, 'w', csv_filename2, current_path)

        # Write the content in the CSV
        for filename in filenames:
            content = get_content_list(filename, exif_data)
            # print(content)
            writeCSV(content, 'a', csv_filename2, current_path)

        if csv_filename2 in os.listdir(current_path):
            print('csv generated')
        else:
            print('csv generation failed')
    except Exception as e:
        print('Error in generateExifDataset: ', e)
    return


# Run if you want to test csv generation
if __name__ == '__main__':
    generateExifDataset()
