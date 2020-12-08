from JDSH import JDSH
import argparse
from easydict import EasyDict as edict
import json
from utils import logger


def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dadaset:{}'.format(config.DATASET))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Alpha:{}'.format(config.alpha))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))


def main():
    parser = argparse.ArgumentParser(description='JDSH')
    parser.add_argument('--Train', default=True, help='train or test', type=bool)
    parser.add_argument('--Config', default='./config/JDSH_MIRFlickr.json', help='Configure path', type=str)
    parser.add_argument('--Dataset', default='MIRFlickr', help='MIRFlickr or NUSWIDE', type=str)
    parser.add_argument('--Checkpoint', default='MIRFlickr_BIT_128.pth', help='checkpoint name', type=str)
    parser.add_argument('--Bit', default=16, help='hash bit', type=int)
    parser.add_argument('--alpha', default=0.5, help='alpha', type=float)
    parser.add_argument('--beta', default=0.1, help='beta', type=float)
    parser.add_argument('--lamada', default=0.4, help='lamada', type=float)
    parser.add_argument('--mu', default=1.4, help='mu', type=float)
    args = parser.parse_args()

    # load basic settings
    with open(args.Config, 'r') as f:
        config = edict(json.load(f))

    # update settings
    config.TRAIN = args.Train
    config.DATASET = args.Dataset
    config.CHECKPOINT = args.Checkpoint
    config.HASH_BIT = args.Bit
    config.alpha = args.alpha
    config.beta = args.beta
    config.lamada = args.lamada
    config.mu = args.mu

    # log
    log = logger()
    log_info(log, config)

    Model = JDSH(log, config)

    if config.TRAIN == False:
        Model.load_checkpoints(config.CHECKPOINT)
        Model.eval()

    else:
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()
            # save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints()


if __name__ == '__main__':
    main()
