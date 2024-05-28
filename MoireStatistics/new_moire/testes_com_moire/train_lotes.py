import sys #
import os #
import tensorflow as tf #
import numpy as np #
import argparse #
from tensorflow.python.client import device_lib #
from os import listdir #
from os.path import join
from PIL import Image #
from sklearn.model_selection import train_test_split #
from mCNN import createModel #
from keras.callbacks import ModelCheckpoint #


# Configurações do modelo e das imagens
config = {
    "width": 512,
    "height": 384,
    "depth": 1,
    "num_classes": 2,
}



def prepare_data_paths(positive_path, negative_path):
    # Subfunção para listar e agrupar arquivos por base de nome
    def list_and_group_files(path):
        try:
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tiff')])
            grouped_files = {}
            for file in files:
                # Estrai a base do nome do arquivo para agrupar componentes correspondentes
                base_name = "_".join(file.split('_')[:-1])
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(file)
            return list(grouped_files.values())
        except Exception as e:
            print(f"Erro ao acessar {path}: {e}")
            return []

    # Caminhos para as subpastas de imagens positivas e negativas
    positive_grouped = list_and_group_files(positive_path)
    negative_grouped = list_and_group_files(negative_path)

    if not positive_grouped and not negative_grouped:
        raise ValueError("Nenhum arquivo válido encontrado nas pastas especificadas.")

    # Combina os grupos de dados de imagens positivas e negativas
    grouped_files = positive_grouped + negative_grouped

    # Divisão dos dados em conjuntos de treinamento e validação
    try:
        train_groups, val_groups = train_test_split(grouped_files, test_size=0.1, random_state=42)
    except ValueError as e:
        print("Erro ao dividir os dados: ", e)
        return [], []

    # Desagrupa os arquivos para retornar a lista plana, necessária para o gerador de dados
    train_files = [item for sublist in train_groups for item in sublist]
    val_files = [item for sublist in val_groups for item in sublist]

    return train_files, val_files




def main(args):

    # Extract paths and epochs from the command line arguments
    positiveImagePath = (args.trainDataPositive)
    negativeImagePath = (args.trainDataNegative)
    numEpochs = (args.epochs)
    batch_size = (args.batch_size) #por default vai ser 32, use sempre um multiplo de 4.




    # Prepare data paths and get train/validation splits
    train_files, val_files = prepare_data_paths(args.trainDataPositive, args.trainDataNegative)


    
    # Create data generators for training and validation sets
    train_generator = image_data_generator(train_files, batch_size=batch_size)
    validation_generator = image_data_generator(val_files, batch_size=batch_size)
    
    
    # Calculate steps per epoch and validation steps based on batch size
    steps_per_epoch = int(np.ceil(len(train_files) / batch_size))
    validation_steps = int(np.ceil(len(val_files) / batch_size))

    
    # Train the model
    model = trainCNNModel(train_generator, steps_per_epoch, validation_generator, validation_steps,config["height"], config["width"], config["depth"], config["num_classes"], args.epochs)

def load_image(file_path):
    """ Carrega uma imagem, a converte para um array NumPy, redimensiona e normaliza para um tamanho uniforme. """
    with Image.open(file_path) as img:
        img_array = np.array(img, dtype=np.float32)
        return img_array / 255.0  # Normaliza para o intervalo 0-1





def image_data_generator(image_files, batch_size=32):
    """ Gera batches de dados com quatro componentes de imagens e seus rótulos. """
    num_samples = len(image_files) // 4  # Considera que cada imagem contribui com 4 componentes
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_files = image_files[offset*4:(offset + batch_size)*4]
            # Ordena os arquivos de forma que a sequência LL, LH, HL, HH seja mantida
            batch_files_sorted = sorted(batch_files, key=lambda x: (x.split('_')[-1].replace('.tiff', ''), x.split('_')[-2]))
            X_LL, X_LH, X_HL, X_HH, Y = [], [], [], [], []
            if len(batch_files_sorted) % 4 != 0:  # Certifica-se de que o lote está completo
                continue
            for i in range(0, len(batch_files_sorted), 4):
                # Carrega as imagens na ordem correta
                x_ll = load_image(batch_files_sorted[i])
                x_lh = load_image(batch_files_sorted[i+1])
                x_hl = load_image(batch_files_sorted[i+2])
                x_hh = load_image(batch_files_sorted[i+3])
                # Adiciona a categoria baseada no prefixo do nome do arquivo
                label = 1 if 'p' in batch_files_sorted[i].split('/')[-1] else 0
                X_LL.append(x_ll)
                X_LH.append(x_lh)
                X_HL.append(x_hl)
                X_HH.append(x_hh)
                Y.append(label)
            # Verifica se a quantidade de componentes corresponde ao tamanho do batch
            if len(X_LL) == batch_size:
                yield ({'input_layer': np.array(X_LL), 'input_layer_1': np.array(X_LH),
                        'input_layer_2': np.array(X_HL), 'input_layer_3': np.array(X_HH)}, np.array(Y))


# Configuração da Estratégia de Múltiplas GPUs
def trainCNNModel(generator, steps_per_epoch, validation_data, validation_steps, height, width, depth, num_classes, num_epochs):

    # Set up the strategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # Create and compile the CNN model
        model = createModel(height, width, depth, num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  
            # Aqui você pode ajustar para: .SGD(learning_rate=0.01, momentum=0.9) ; .Adam(learning_rate=0.001) ;     .RMSprop(learning_rate=0.001) ; .Adagrad(learning_rate=0.01) ; e alterar a taxa de aprendizado de acordo com o que quiser.
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Create a directory for model checkpoints
        checkPointFolder = 'checkPoint'
        if not os.path.exists(checkPointFolder):
            os.makedirs(checkPointFolder)
        checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.keras'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]

        # Training the model
        model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks_list)

    # Evaluate the model on the validation data
    score, acc = model.evaluate(validation_data, steps=validation_steps, verbose=1)
    print(f"Validation score: {score}, Accuracy: {acc}")

    # Save the final model
    model.save('moirePattern3CNN_.keras')
    
    return model



def parse_arguments(argv):

    parser = argparse.ArgumentParser(description="Script to train a CNN to detect Moiré patterns in images.")

    # Adding arguments for the paths to the datasets for positive and negative examples
    parser.add_argument('trainDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('trainDataNegative', type=str, help='Directory with transformed negative (Normal) images.')
    
    # Adding an argument for the number of training epochs
    parser.add_argument('epochs', type=int, help='Number of epochs for training.')
    
    # Batch size for training.
    parser.add_argument('batch_size', type=int, default=32, help='Batch size for training.')

    # Parse and return the arguments
    return parser.parse_args(argv)


if __name__ == '__main__':

    try:
        # Check and display the number of available GPUs.
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", num_gpus)

        # Optionally, list all devices detected by TensorFlow. This includes CPUs and GPUs.
        local_devices = device_lib.list_local_devices()
        print("Local devices detected:\n", local_devices)

        # Parsing command line arguments required for the main function.
        parsed_args = parse_arguments(sys.argv[1:])
        
        # Execute the main function with the parsed arguments.
        main(parsed_args)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

