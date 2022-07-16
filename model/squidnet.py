import torch
import torch.nn as nn
from model.encoder import VSMMApool, VQMMApool, NVLDModel
from model.layers import JointSpaceEmbedding
from model.cmp import ConditionalMomentPrediction

import logging
import pdb
logger = logging.getLogger(__name__)


class SQuiDNet(nn.Module):
    def __init__(self, config, vid_dim=4352, text_dim=768, hidden_dim=768, lw_st_ed=0.01, lw_vid=0.005, loss_measure="moment_video"):

        super(SQuiDNet, self).__init__()
        self.config = config

        #  related configs
        self.lw_st_ed = lw_st_ed
        self.lw_vid = lw_vid
        self.loss_measure = loss_measure

        # VSMMA encoder (video-subtitle modality matching attention)
        self.MMAencoder = VSMMApool(config, vid_dim=vid_dim, text_dim=text_dim, hidden_dim=hidden_dim)

        # VQMMA_Plus encoder
        self.nvld_query_weight = NVLDModel(config.netvlad_config)
        self.JS_emb = JointSpaceEmbedding(config.joint_emb_config, hidden_dim*3) 
        
        # VQMMA encoder (video-query modality matching attention)
        self.VQMMA = VQMMApool(hidden_dim)

        # CMP (conditional moment prediction)
        self.CMP = ConditionalMomentPrediction(config.moment_prediction_config, config.joint_emb_config, hidden_dim)
        
        # CE (cross entropy loss)
        self.CE = nn.CrossEntropyLoss(reduction="mean")

        ## video_prediction
        if self.loss_measure == "moment_video":
            self.score_ce = nn.CrossEntropyLoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                #print("nn.Linear, nn.Embedding: ", module)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)


    def vs_fusion_with_nvld_query(self, vid_feat, sub_feat, query_feat):
        query_weights = self.nvld_query_weight(query_feat)
        vs_feat = torch.einsum("bld,b->bld", vid_feat, query_weights['vid']) + torch.einsum("bld,b->bld", sub_feat, query_weights['sub']) 
        return vs_feat
    
    def VQMMA_Plus(self, vid_feat, sub_feat, query_feat, vid_mask, query_mask):

        vs_feat = self.vs_fusion_with_nvld_query(vid_feat, sub_feat, query_feat)

        vq_feat = self.VQMMA(vs_feat, query_feat, vid_mask, query_mask)

        res_feat = self.JS_emb(features=vq_feat, feat_mask=vid_mask)
        
        final_feat = torch.cat([vq_feat,res_feat], dim=2)    

        return final_feat, res_feat
    
    def query_repeat(self, batch, query_feat, vid_feat):
        query_batch = query_feat.shape[0]
        vid_batch, vid_len = vid_feat.shape[:2]
        tot_nmr_bmr_num = int(vid_batch / query_batch) # check total nmr bmr videos per query

        # query expension for contrastive learninig (BMR vs NMR)        
        query_feat = torch.repeat_interleave(query_feat, tot_nmr_bmr_num, dim=0)
        query_mask = torch.repeat_interleave(batch["query"]["feat_mask"], tot_nmr_bmr_num, dim=0)

        return query_feat, query_mask, tot_nmr_bmr_num

    def get_pred_from_raw_query(self, batch):

        # get query feature and subtitle-matched video feature
        query_feature = self.MMAencoder.query_enc(batch)
        vsMMA_feature = self.MMAencoder.VSMMA(batch)
        query_batch = query_feature.shape[0]

        # video features include NMR and BMR
        sample_key = list(vsMMA_feature.keys())[0]
        sub_matched_vid_feature = vsMMA_feature['vid']
        vid_matched_sub_feature = vsMMA_feature['sub'] # subtitlte features are also available to use
        vid_batch, vid_len = sub_matched_vid_feature.shape[:2]
        vid_mask = batch["vid"]["feat_mask"]
        
        query_feature, query_mask, tot_nmr_bmr_num = self.query_repeat(batch, query_feature, sub_matched_vid_feature)
    
        # VQMMA Plus
        final_feat, res_feat = self.VQMMA_Plus(sub_matched_vid_feature, vid_matched_sub_feature, query_feature, vid_mask, query_mask)

        # CMP
        start_time_distribution, end_time_distribution = self.CMP(final_feat, res_feat, vid_mask)
        start_time_distribution = start_time_distribution.view(query_batch, tot_nmr_bmr_num, vid_len)
        end_time_distribution = end_time_distribution.view(query_batch, tot_nmr_bmr_num, vid_len)

        video_prediction_score = None
        if self.loss_measure == "moment_video":
            video_prediction_score = self.CMP.video_prediction(final_feat)
            video_prediction_score = video_prediction_score.view(query_batch, tot_nmr_bmr_num)

        return video_prediction_score, start_time_distribution, end_time_distribution


    def moment_level_debiasing_loss(self, start_time_distribution, end_time_distribution, st_ed_indices, is_positive):

        bs , shared_video_num , video_len = start_time_distribution.size()

        start_time_distribution = start_time_distribution.view(bs,-1)
        end_time_distribution = end_time_distribution.view(bs,-1)

        loss_st = self.CE(start_time_distribution, st_ed_indices[:, 0])
        loss_ed = self.CE(end_time_distribution, st_ed_indices[:, 1])
        moment_debiasing_loss = loss_st + loss_ed

        return moment_debiasing_loss


    def forward(self,batch):
        video_prediction_score, start_time_distribution , end_time_distribution = self.get_pred_from_raw_query(batch)
        moment_debiasing_loss, vid_loss = 0, 0
        moment_debiasing_loss = self.moment_level_debiasing_loss(start_time_distribution, end_time_distribution, batch["st_ed_indices"], batch["is_positive"])
        moment_debiasing_loss = self.lw_st_ed * moment_debiasing_loss

        if self.loss_measure == "moment_video":
            vid_label = batch["st_ed_indices"].new_zeros(video_prediction_score.size()[0])
            vid_loss = self.score_ce(video_prediction_score, vid_label)
            vid_loss = self.lw_vid * vid_loss


        loss = moment_debiasing_loss + vid_loss
        return loss, {"moment_debiasing_loss": float(moment_debiasing_loss), "video_prediction_loss": float(vid_loss), "loss_total": float(loss)}
        #return loss




