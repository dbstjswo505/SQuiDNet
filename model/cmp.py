import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)
from model.modeling_utils import mask_logits
from model.layers import JointSpaceEmbedding
import pdb

class Conv1D(nn.Module):
    def __init__(self, config):
        super(Conv1D, self).__init__()

        self.moment_predictor = nn.Sequential(nn.Conv1d(**config.conv_cfg_1), nn.ReLU(), nn.Conv1d(**config.conv_cfg_2))

    def forward(self, features, video_mask):
        score = self.moment_predictor(features).squeeze(1)
        score = mask_logits(score, video_mask)
        return score

class ConditionalMomentPrediction(nn.Module):

    def __init__(self, config, joint_emb_config, hidden_dim):
        super(ConditionalMomentPrediction, self).__init__()

        self.start_JSemb = JointSpaceEmbedding(joint_emb_config, hidden_dim * 4)
        self.conditional_end_JSemb = JointSpaceEmbedding(joint_emb_config, hidden_dim * 2)

        self.start_moment_prediction = Conv1D(config)
        self.end_moment_prediction = Conv1D(config)
        self.video_predictor = nn.Sequential(nn.Linear(**config.linear_cfg_1),nn.ReLU(),nn.Linear(**config.linear_cfg_2),nn.ReLU(),nn.Linear(**config.linear_cfg_3))

    def video_prediction(self, final_feat):
        candidate_video_feature, _ = torch.max(final_feat, dim=1)
        
        video_prediction_score = self.video_predictor(candidate_video_feature.squeeze(1))
        
        return video_prediction_score

    def forward(self, final_feat, res_feat, vid_mask):
        
        start_features = self.start_JSemb(features=final_feat, feat_mask=vid_mask)
        # get start time information as prior knowldege
        end_features = self.conditional_end_JSemb(features=torch.cat([res_feat, start_features], dim=2), feat_mask=vid_mask)

        start_feature_ = torch.transpose(start_features, 1, 2)
        end_feature_ = torch.transpose(end_features, 1, 2)

        start_moment_distribution = self.start_moment_prediction(features=start_feature_, video_mask=vid_mask)
        end_moment_distribution = self.end_moment_prediction(features=end_feature_, video_mask=vid_mask)

        return start_moment_distribution , end_moment_distribution


