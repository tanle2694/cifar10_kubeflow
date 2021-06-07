import argparse
import torch
import json
from cifarlibs.evaluation.evaluator import CifarEvaluator
import logging
import cifarlibs.utils.logging_handler
torch.manual_seed(43)
import time
logger = logging.getLogger(__name__)


def get_pipeline_args():
    parser = argparse.ArgumentParser("Training step")

    parser.add_argument('--batch_size', type=int, default=32, help="batch size use when training")
    parser.add_argument("--input_data", type=str,
                        default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pytorch-cifar/data",
                        help="Directory include data")
    parser.add_argument("--model_path", type=str, help="best model", default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pipeline_source/training_step/checkpoints/model_best.pth")
    parser.add_argument("--loss_name", type=str, help="loss_name", default="cross_entropy")
    parser.add_argument("--eval_result", type=str, help="File to write eval result", default='./save_result/result.txt')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logger.info("Getting args")
    args = get_pipeline_args()

    logger.info("Setup evaluator")
    evaluator = CifarEvaluator(input_data_dir=args.input_data,
                           batch_size=args.batch_size,
                           loss_name=args.loss_name,
                           model=args.model_path,
                           )

    logger.info("Start evaluation")
    start = time.time()
    avg_loss, avg_acc = evaluator.start_evaluate()
    logger.info("Finished Evaluation step")
    logger.info(f"Time to training {time.time() - start}")
    logger.info(f"Acc: {avg_acc} Loss: {avg_loss}")
    evaluator.save_result(avg_loss, avg_acc, args.eval_result)
