from session import JDSH
import argparse
import config


def main():
    parser = argparse.ArgumentParser(description='JDSH')
    parser.add_argument('--Train', default=True, help='train or test', type=bool)
    parser.add_argument('--Dataset', default='NUSWIDE', help='MIRFlickr or NUSWIDE', type=str)
    parser.add_argument('--Checkpoint', default='MIRFlickr_BIT_128.pth', help='checkpoint name', type=str)
    parser.add_argument('--Bit', default=128, help='hash bit', type=int)
    parser.add_argument('--alpha', default=0.4, help='alpha', type=float)
    parser.add_argument('--beta', default=0.3, help='beta', type=float)
    parser.add_argument('--lamada', default=0.3, help='lamada', type=float)
    parser.add_argument('--mu', default=1.47, help='mu', type=float)

    args = parser.parse_args()

    config.TRAIN = args.Train
    config.DATASET = args.Dataset
    config.CHECKPOINT = args.Checkpoint
    config.HASH_BIT = args.Bit

    config.alpha = args.alpha
    config.beta = args.beta
    config.lamada = args.lamada
    config.mu = args.mu

    Model = JDSH()
    Model.log_info()

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