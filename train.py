import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.config import SharedOpt
from model.squidnet import SQuiDNet
from loader import SQDataset
from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter,load_config
from utils.model_utils import count_parameters, set_cuda, vcmr_collate
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
import pdb


def train_epoch(model, train_loader, optimizer, opt, epoch_i ,training=True):
    model.train(mode=training)

    # init meters
    loss_meters = OrderedDict(moment_debiasing_loss=AverageMeter(), video_prediction_loss=AverageMeter(), loss_total=AverageMeter())

    num_training = len(train_loader)
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training", total=num_training):
        global_step = epoch_i * num_training + batch_idx

        # continue
        model_inputs = set_cuda(batch["model_inputs"], opt.device)
        loss, loss_dict = model(model_inputs)

        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if total_norm > opt.grad_clip:
                    print("clipping gradient: {} with coefficient as {}".format(total_norm, opt.grad_clip / total_norm))

            optimizer.step()
            opt.writer.add_scalar("Train/LR_top", float(optimizer.param_groups[0]["lr"]), global_step)
            opt.writer.add_scalar("Train/LR_pretrain", float(optimizer.param_groups[-1]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))


    if training:
        to_write = opt.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),epoch=epoch_i,loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def build_optimizer(model, opts):
    param_optimizer = [(n, p) for n, p in model.named_parameters() if (n.startswith('encoder') or n.startswith('query_weight')) and p.requires_grad ]
    param_top = [(n, p) for n, p in model.named_parameters() if  ( not n.startswith('encoder') and not n.startswith('query_weight'))  and p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_top if not any(nd in n for nd in no_decay)], 'weight_decay': opts.wd},
        {'params': [p for n, p in param_top if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': opts.lr_mul * opts.lr, 'weight_decay': opts.wd},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': opts.lr_mul * opts.lr, 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.lr)
    return optimizer


def train(model, train_dataset, train_eval_dataset, val_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        #assert len(opt.device_ids) == 1
        if len(opt.device_ids) > 1:
            #logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    train_loader = DataLoader(train_dataset, collate_fn=vcmr_collate, batch_size=opt.batch, num_workers=opt.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, collate_fn=vcmr_collate, batch_size=opt.batch, num_workers=opt.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    # Prepare optimizer
    optimizer = build_optimizer(model, opt)

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = 0 if opt.no_eval_untrained else -1
    eval_tasks = opt.eval_tasks 
    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(opt.data_name, opt.eval_type, "_".join(eval_tasks))
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i >= 0:

            train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True)

        global_step = (epoch_i + 1) * len(train_loader)
        if epoch_i % opt.eval_epoch_num == 0 or epoch_i == opt.n_epoch - 1 or epoch_i == start_epoch:
            with torch.no_grad():
                train_epoch(model, train_eval_loader, optimizer, opt, epoch_i, training=False)

                metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, val_dataset, opt, save_submission_filename, tasks=eval_tasks, max_after_nms=100)
            to_write = opt.eval_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),epoch=epoch_i, eval_metrics_str=json.dumps(metrics_no_nms))
            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            # logger.info("query_type_acc \n{}".format(pprint.pformat(query_type_acc_dict, indent=4)))
            logger.info("metrics_no_nms {}".format(pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
            logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))

            # metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
            metrics = metrics_no_nms
            # early stop/ log / save model
            for task_type in ["SVMR", "VCMR"]:
                if task_type in metrics:
                    task_metrics = metrics[task_type]
                    for iou_thd in [0.5, 0.7]:
                        opt.writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd), {k: v for k, v in task_metrics.items() if str(iou_thd) in k}, global_step)

            task_type = "VR"
            if task_type in metrics:
                task_metrics = metrics[task_type]
                opt.writer.add_scalars("Eval/{}".format(task_type), {k: v for k, v in task_metrics.items()}, global_step)

            # use the most strict metric available
            stop_metric_names = ["r1"] if opt.task == "VR" else ["0.5-r1", "0.7-r1"]
            stop_score = sum([metrics[opt.task][e] for e in stop_metric_names])

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                if len(opt.device_ids) > 1:
                    checkpoint = {"model": model.state_dict(), "model_cfg": model.module.config, "epoch": epoch_i}
                else:
                    checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
                torch.save(checkpoint, opt.ckpt_filepath)

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write("Early Stop at epoch {}".format(epoch_i))
                    logger.info("Early stop at {} with {} {}"
                                .format(epoch_i, " ".join([opt.task] + stop_metric_names), prev_best_score))
                    break

    opt.writer.close()


def train_squid():
    logger.info("setup opt configuration...")
    opt = SharedOpt().parse()
    # Fix seed
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensuer the cuda is available
    torch.cuda.manual_seed_all(seed)

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str}: epch {epoch:03d} loss {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str}: epch {epoch:03d} metrics {eval_metrics_str}\n"

    data_config = load_config(opt.data_config)
    train_dataset = SQDataset(data_type="train", config=data_config, neg_bmr_pred_num=opt.neg_bmr_pred_num, bmr_allowance=opt.bmr_allowance)
    train_eval_dataset = SQDataset(data_type="val", config=data_config, max_vid_len=opt.max_vid_len, max_query_len=opt.max_query_len, neg_bmr_pred_num=opt.neg_bmr_pred_num, bmr_allowance=opt.bmr_allowance)
    eval_dataset = SQDataset(data_type=opt.eval_type, config=data_config, max_vid_len=opt.max_vid_len, max_query_len=opt.max_query_len, is_val=True, max_vcmr_video=opt.max_vcmr_video)

    model_config = load_config(opt.model_config)
    model = SQuiDNet(model_config, vid_dim=opt.vid_dim, text_dim=opt.text_dim, hidden_dim=opt.hidden_dim, lw_vid=opt.lw_vid, lw_st_ed=opt.lw_st_ed, loss_measure=opt.loss_measure)

    print(model)
    count_parameters(model)

    logger.info("Start Training...")
    train(model, train_dataset, train_eval_dataset, eval_dataset, opt)
    return opt.results_dir, opt.eval_type


if __name__ == '__main__':
    model_dir, eval_type  = train_squid()

