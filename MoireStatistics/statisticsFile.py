def joinPositiveNegativeStatisticsDataV2(filename_positive, filename_negative, filename_both):
    try:
        if (filename_positive.split('.')[-1] != 'txt' or filename_negative.split('.')[-1] != 'txt'
            or filename_both.split('.')[-1] != 'txt'):
            print('Error: Invalid file format. File must be a .txt file!')
            exit(1)


        file_positive = open(filename_positive, "r")
        file_negative = open(filename_negative, "r")
        file_both = open(filename_both, "w")

        content_positive = file_positive.read()
        content_negative = file_negative.read()

        file_both.write(content_positive)
        file_both.write('\n\n')
        file_both.write(content_negative)

        file_positive.close()
        file_negative.close()
        file_both.close()
    except Exception as e:
        print("Error in joinPositiveNegativeStatisticsDataV2: ", e)
        exit(1)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Junta o conteúdo de dois arquivos")
    parser.add_argument("filename_positive", type=str, help="Caminho para o arquivo.")
    parser.add_argument("filename_negative", type=str, help="Caminho para o arquivo.")
    parser.add_argument("filename_both", type=str, help="Caminho para o arquivo.")
    args = parser.parse_args()

    statistics(args.filename_positive, args.filename_negative, args.filename_both)