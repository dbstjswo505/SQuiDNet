from utils.temporal_nms import temporal_non_maximum_suppression
from collections import defaultdict


def get_submission_top_n(submission, top_n=100):
    def get_prediction_top_n(list_dict_predictions, top_n):
        top_n_res = []
        for e in list_dict_predictions:
            e["predictions"] = e["predictions"][:top_n]
            top_n_res.append(e)
        return top_n_res

    top_n_submission = dict(vid2idx=submission["vid2idx"], )
    for k in submission:
        if k != "vid2idx":
            top_n_submission[k] = get_prediction_top_n(submission[k], top_n)
    return top_n_submission



def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is{
        "desc": str,
        "desc_id": int,
        "predictions": list(sublist)  # each sublist is
            [video_idx (int), st (float), ed(float), score (float)], video_idx could be different
    }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        e["predictions"] = filter_vcmr_by_nms(e["predictions"],
                                              nms_threshold=nms_thd,
                                              max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6,
                       max_before_nms=1000, max_after_nms=100, score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    Returns:

    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(pred[1:])  # [st (float), ed(float), score (float)]

    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = \
            temporal_non_maximum_suppression(grouped_preds, nms_threshold=nms_threshold)

    predictions_after_nms = []
    for video_idx, grouped_preds in predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            pred = [video_idx] + pred  # [video_idx (int), st (float), ed(float), score (float)]
            predictions_after_nms.append(pred)

    # ranking happens across videos
    predictions_after_nms = sorted(predictions_after_nms,
                                   key=lambda x: x[score_col_idx],
                                   reverse=True)[:max_after_nms]  # descending order
    return predictions_after_nms
