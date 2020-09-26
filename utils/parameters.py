# hrb_input parameters
IMG_DIR = '../../../images/'
HEIGHT = 292 * 2
WIDTH = 196 * 2

P1 = 0.3
P2 = 0.6
P3 = 0.9

TRAIN = 'dataset/train.txt'
VAL = 'dataset/validation.txt'
TEST = 'dataset/test.txt'

# training parameters
LR = 1e-4

BATCH_SIZE = 60
EPOCHS = 60
EARLY_STOPPING_EPOCHS = 10

LR_STEPS = [10, 20, 30, 40]
LR_GAMMA = 0.5

PRINT_FREQ = 200

MODEL_PATH = 'model/model.pt'
