
TRAIN = True


DATASET = 'MIRFlickr'


if DATASET == 'MIRFlickr':

    #
    alpha = 0.5
    beta = 0.1
    lamb = 0.4

    mu = 1.47

    INTRA = 0.1

    LR_IMG = 0.01
    LR_TXT = 0.01

    MIN = -0.52
    MAX = 0.68

    ALPHA = 2
    LOC_LEFT = -0.49  # 015904552551
    SCALE_LEFT = 0.019

    BETA = 4
    LOC_RIGHT = -0.49
    SCALE_RIGHT = 0.088

    L1 = 0.1
    L2 = 0.2

    LABEL_DIR = 'dataset/Flickr/mirflickr25k-lall.mat'
    TXT_DIR = 'dataset/Flickr/mirflickr25k-yall.mat'
    IMG_DIR = 'dataset/Flickr/mirflickr25k-iall.mat'


if DATASET == 'NUSWIDE':


    alpha = 0.4
    beta = 0.3
    lamb = 0.3

    mu = 1.47

    INTRA = 0.1

    LR_IMG = 0.001
    LR_TXT = 0.01

    MAX=0.68
    MIN=-0.52

    ALPHA = 2
    LOC_LEFT = -0.62 # 015904552551
    SCALE_LEFT = 0.0128


    L1 = 0.2
    L2 = 0.2


    LABEL_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-lall.mat'
    TXT_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-yall.mat'
    IMG_DIR = 'dataset/NUS-WIDE/nus-wide-tc10-iall.mat'



HASH_BIT = 128

BATCH_SIZE = 32

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 8
EPOCH_INTERVAL = 2
NUM_EPOCH = 300
EVAL_INTERVAL = 40

MODEL_DIR = './checkpoint'
CHECKPOINT = 'flikcr_128bit.pth'

