import os
import pprint
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from model.squidnet import SQuiDNet
from loader import SQDataset
from config.config import TestOpt
from utils.inference_utils  import get_submission_top_n, post_processing_vcmr_nms
from utils.basic_utils import save_json, load_config
from utils.tensor_utils import find_max_triples_from_upper_triangle_product
from standalone_eval.eval import eval_retrieval
from utils.model_utils import set_cuda, vcmr_collate, N_Infinite
import logging
from time import time
import pdb
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def svmr_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, ann_info, vid2idx, min_pred_l, max_pred_l, max_before_nms):
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in ann_info]
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)
    valid_prob_mask = generate_min_max_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(st_ed_prob_product, top_n=max_before_nms, prob_thd=None)
    for i, q_vid_name in tqdm(enumerate(query_vid_names), desc="[SVMR]", total=len(query_vid_names)):
        q_m = ann_info[i]
        video_idx = vid2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:, 1] += 1
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * 1.5
        cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(desc_id=q_m["desc_id"], desc=q_m["desc"], predictions=cur_ranked_predictions)
        svmr_res.append(cur_query_pred)
    return svmr_res


def generate_min_max_mask(array_shape, min_l, max_l):
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32) 
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask

def compute_query2vid(model, eval_dataset, opt, max_before_nms=200, max_vcmr_video=100, tasks=("SVMR",)):
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    is_svmr = "SVMR" in tasks

    vid2idx = eval_dataset.vid2idx

    model.eval()
    query_eval_loader = DataLoader(eval_dataset, collate_fn=vcmr_collate, batch_size=opt.eval_query_batch, num_workers=opt.num_workers, shuffle=False, pin_memory=True)

    n_total_query = len(eval_dataset)
    query_batch_size = opt.eval_query_batch

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    if is_vr :
        sorted_q2c_indices = np.empty((n_total_query, max_vcmr_video), dtype=np.int)
        sorted_q2c_scores = np.empty((n_total_query, max_vcmr_video), dtype=np.float32)
    if is_svmr:
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_vid_len), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_vid_len), dtype=np.float32)

    ann_info = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        ann_info.extend(batch["annotation"])
        model_inputs = set_cuda(batch["model_inputs"], opt.device)

        if opt.device.type == "cuda":
            model_inputs = set_cuda(batch["model_inputs"], opt.device)
        else:
            model_inputs = batch["model_inputs"]

        if len(opt.device_ids) > 1:
            video_similarity_score, begin_score_distribution, end_score_distribution = model.module.get_pred_from_raw_query(model_inputs)
        else:
            video_similarity_score, begin_score_distribution, end_score_distribution = model.get_pred_from_raw_query(model_inputs)

        if is_svmr:
            _svmr_st_probs = begin_score_distribution[:, 0]
            _svmr_ed_probs = end_score_distribution[:, 0]

            _svmr_st_probs = F.softmax(_svmr_st_probs, dim=-1)
            _svmr_ed_probs = F.softmax(_svmr_ed_probs, dim=-1)

            svmr_gt_st_probs[idx*query_batch_size : (idx+1)*query_batch_size] = _svmr_st_probs.cpu().numpy()
            svmr_gt_ed_probs[idx*query_batch_size : (idx+1)*query_batch_size] = _svmr_ed_probs.cpu().numpy()

        _vcmr_st_prob = begin_score_distribution[:, 1:]
        _vcmr_ed_prob = end_score_distribution[:, 1:]

        if not (is_vr or is_vcmr):
            continue
        video_similarity_score = video_similarity_score[:, 1:] # first element holds ground-truth information
        _query_context_scores = torch.softmax(video_similarity_score,dim=1)
        _sorted_q2c_scores, _sorted_q2c_indices = torch.topk(_query_context_scores, max_vcmr_video, dim=1, largest=True)

        if is_vr:
            sorted_q2c_indices[idx*query_batch_size : (idx+1)*query_batch_size] = _sorted_q2c_indices.cpu().numpy()
            sorted_q2c_scores[idx*query_batch_size : (idx+1)*query_batch_size] = _sorted_q2c_scores.cpu().numpy()

        if not is_vcmr:
            continue

        _st_probs = F.softmax(_vcmr_st_prob, dim=-1)  # (query_batch, video_corpus, vid_len)
        _ed_probs = F.softmax(_vcmr_ed_prob, dim=-1)

        row_indices = torch.arange(0, len(_st_probs), device=opt.device).unsqueeze(1)
        _st_probs = _st_probs[row_indices, _sorted_q2c_indices] 
        _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]

        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs, _sorted_q2c_scores, _ed_probs)

        valid_prob_mask = generate_min_max_mask(_st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)

        _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(_st_ed_scores.device)

        _n_q  = _st_ed_scores.shape[0]

        _flat_st_ed_scores = _st_ed_scores.reshape(_n_q, -1)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = torch.sort(_flat_st_ed_scores, dim=1, descending=True)

        flat_st_ed_sorted_scores[idx*query_batch_size : (idx+1)*query_batch_size] = _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx*query_batch_size : (idx+1)*query_batch_size] = _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()

    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(enumerate(zip(sorted_q2c_scores, sorted_q2c_indices)), desc="[VR]", total=n_total_query):
            vr_predictions = []
            max_vcmr_vid_name_pool = ann_info[i]["max_vcmr_vid_name_list"]
            for j, (v_score, v_name_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = vid2idx[max_vcmr_vid_name_pool[v_name_idx]]
                vr_predictions.append([video_idx, 0, 0, float(v_score)])
            cur_pred = dict(desc_id=ann_info[i]["desc_id"], desc=ann_info[i]["desc"], predictions=vr_predictions)
            vr_res.append(cur_pred)

    svmr_res = []
    if is_svmr:
        svmr_res = svmr_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, ann_info, vid2idx, min_pred_l=opt.min_pred_l, max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms)


    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),desc="[VCMR]", total=n_total_query):
            video_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(_flat_st_ed_scores_sorted_indices, shape=(max_vcmr_video, opt.max_vid_len, opt.max_vid_len))
            video_indices = sorted_q2c_indices[i, video_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(np.float32) * 1.5
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * 1.5 + 1.5
            vcmr_predictions = []
            max_vcmr_vid_name_pool = ann_info[i]["max_vcmr_vid_name_list"]
            for j, (v_score, v_name_idx) in enumerate(zip(_flat_st_ed_sorted_scores, video_indices)):  # videos
                video_idx = vid2idx[max_vcmr_vid_name_pool[v_name_idx]]
                vcmr_predictions.append([video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]), float(v_score)])
            cur_pred = dict(desc_id=ann_info[i]["desc_id"], desc=ann_info[i]["desc"], predictions=vcmr_predictions)
            vcmr_res.append(cur_pred)

    res = dict(VCMR=vcmr_res, SVMR=svmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}


def get_eval_res(model, eval_dataset, opt, tasks):
    """compute and save query and video proposal embeddings"""

    eval_res = compute_query2vid(model, eval_dataset, opt, max_before_nms=opt.max_before_nms, max_vcmr_video=opt.max_vcmr_video, tasks=tasks)
    eval_res["vid2idx"] = eval_dataset.vid2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {
    "SVMR": post_processing_vcmr_nms,
    "VCMR": post_processing_vcmr_nms
}


def eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=("VCMR","SVMR","VR"), max_after_nms=100):
    model.eval()
    logger.info("Computing scores")
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks)

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    if opt.eval_type == "val":  # since test_public has no GT
        metrics = eval_retrieval(eval_submission, eval_dataset.query_data, iou_thds=IOU_THDS, match_number=True, verbose=False, use_desc_type=opt.data_name == "tvr")
        save_metrics_path = submission_path.replace(".json", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]


    if opt.nms_thd != -1:
        logger.info("Performing nms with nms_thd {}".format(opt.nms_thd))
        eval_submission_after_nms = dict(vid2idx=eval_submission_raw["vid2idx"])
        if "VR" in eval_submission_raw:
            eval_submission_after_nms["VR"] = eval_submission_raw["VR"]
        for k, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if k in eval_submission_raw:
                eval_submission_after_nms[k] = nms_func(eval_submission_raw[k], nms_thd=opt.nms_thd, max_before_nms=opt.max_before_nms, max_after_nms=max_after_nms)

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)
        if opt.eval_type == "val":
            metrics_nms = eval_retrieval(eval_submission_after_nms, eval_dataset.query_data, iou_thds=IOU_THDS, match_number=True, verbose=False)
            save_metrics_nms_path = submission_nms_path.replace(".json", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    model = SQuiDNet(loaded_model_cfg, vid_dim=opt.vid_dim, text_dim=opt.text_dim, hidden_dim=opt.hidden_dim, loss_measure=opt.loss_measure)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        assert len(opt.device_ids) == 1
        if len(opt.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOpt().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    data_config = load_config(opt.data_config)
    eval_dataset = SQDataset(data_type=opt.eval_type, config=data_config, max_vid_len=opt.max_vid_len, max_query_len=opt.max_query_len, is_val=True, max_vcmr_video=opt.max_vcmr_video)

    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}.json".format(opt.data_name, opt.eval_type, opt.eval_id, "_".join(opt.tasks))
    print(save_submission_filename)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
