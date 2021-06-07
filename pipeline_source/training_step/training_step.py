import argparse
import torch
import json
from cifarlibs.training.trainer import CifarTrainer
import logging
import cifarlibs.utils.logging_handler
torch.manual_seed(43)
import time
logger = logging.getLogger(__name__)


def get_pipeline_args():
    parser = argparse.ArgumentParser("Training step")

    parser.add_argument('--batch_size', type=int, default=32, help="batch size use when training")
    parser.add_argument('--lr', type=float, default=0.01, help="init learningrate")
    parser.add_argument("--use_random_crop", type=bool, default=True, help="use random crop augmentation")
    parser.add_argument("--use_horizon_flip", type=bool, default=True, help="use random horizontal flip augmentation")
    parser.add_argument("--input_data", type=str,
                        default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pytorch-cifar/data",
                        help="Directory include data")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--loss_name", type=str, default="cross_entropy", choices=["cross_entropy"])
    parser.add_argument("--weight_decay", type=float, default=5e-4, )
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
    parser.add_argument("--monitor_spec", type=str, default="acc_max")
    parser.add_argument("--nb_earlystop", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")
    parser.add_argument("--tensorboard_path", type=str, default="/tmp/tensorboard")
    parser.add_argument("--tune_hp", type=str, help="dict str result of hyper parameters tunning step")

    args = parser.parse_args()

    if args.tune_hp:
        tune_hp_json = json.loads(args.tune_hp)
        temp_args = argparse.Namespace()
        temp_args.__dict__.update(tune_hp_json)
        args = parser.parse_args(namespace=temp_args)

    return args


if __name__ == "__main__":
    logger.info("Getting args")
    args = get_pipeline_args()

    logger.info("Setup trainer")
    trainer = CifarTrainer(input_data_dir=args.input_data,
                           epochs=args.epoch,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           use_random_crop=args.use_random_crop,
                           use_horizon_flip=args.use_horizon_flip,
                           loss_name=args.loss_name,
                           weight_decay=args.weight_decay,
                           optimizer_algo=args.optimizer,
                           monitor_spec=args.monitor_spec,
                           nb_earlystop=args.nb_earlystop,
                           checkpoint_path=args.checkpoint_path,
                           tensorboard_path=args.tensorboard_path
                           )
    logger.info("Start training")
    start = time.time()
    trainer.start_train()
    logger.info("Finished training step")
    logger.info(f"Time to training {time.time() - start}")