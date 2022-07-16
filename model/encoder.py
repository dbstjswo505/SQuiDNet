"""
Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""

import torch
import logging
from torch import nn
logger = logging.getLogger(__name__)

try:
  import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
  BertLayerNorm = torch.nn.LayerNorm

from model.transformer.bert import BertEncoder
from model.layers import (NetVLAD, LinearLayer)
from model.transformer.bert_embed import (BertEmbeddings)
from utils.model_utils import mask_logits
import torch.nn.functional as F
import pdb


class TransformerBaseModel(nn.Module):
    """
    Base Transformer model
    """
    def __init__(self, config):
        super(TransformerBaseModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)


    def forward(self,features,position_ids,token_type_ids,attention_mask):
        # embedding layer
        embedding_output = self.embeddings(token_type_ids=token_type_ids,
                                           inputs_embeds=features,
                                           position_ids=position_ids)

        encoder_outputs = self.encoder(embedding_output, attention_mask)

        sequence_output = encoder_outputs[0]

        return sequence_output

class MMA(nn.Module):
    """`
        Modality Matching Attention
    """

    def __init__(self, config, img_dim, text_dim, hidden_dim):
        super(MMA, self).__init__()
        self.img_linear = LinearLayer(in_hsz=img_dim, out_hsz=hidden_dim)
        self.text_linear = LinearLayer(in_hsz=text_dim, out_hsz=hidden_dim)
        self.transformer = TransformerBaseModel(config)
        self.vid_num = config.max_position_embeddings


    def forward(self, vid_features, vid_position_ids, vid_token_type_ids, vid_attention_mask,
                text_features,text_position_ids,text_token_type_ids,text_attention_mask):

        transformed_im = self.img_linear(vid_features)
        transformed_text = self.text_linear(text_features)

        transformer_input_feat = torch.cat((transformed_im,transformed_text),dim=1)
        transformer_input_feat_pos_id = torch.cat((vid_position_ids,text_position_ids),dim=1)
        transformer_input_feat_token_id = torch.cat((vid_token_type_ids,text_token_type_ids),dim=1)
        transformer_input_feat_mask = torch.cat((vid_attention_mask,text_attention_mask),dim=1)

        output = self.transformer(features=transformer_input_feat,
                                  position_ids=transformer_input_feat_pos_id,
                                  token_type_ids=transformer_input_feat_token_id,
                                  attention_mask=transformer_input_feat_mask)
        
        return torch.split(output, self.vid_num, dim=1)


class Encoder(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = LinearLayer(in_hsz=input_dim, out_hsz=hidden_dim)
        self.transformer = TransformerBaseModel(config)
    def forward(self, features, position_ids, token_type_ids, attention_mask):
        transformed_features = self.linear(features)
        output = self.transformer(features=transformed_features,position_ids=position_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        return output


class VSMMApool(nn.Module):
    def __init__(self, config, vid_dim=4352, text_dim=768, hidden_dim=768):
        super(VSMMApool, self).__init__()
        self.vsMMA = MMA(config=config.joint_emb_config, img_dim=vid_dim, text_dim=text_dim, hidden_dim=hidden_dim)
        self.text_enc = Encoder(config=config.query_enc_config, input_dim=text_dim, hidden_dim=hidden_dim)

    def query_enc(self, batch):

        query_output = self.text_enc(features=batch["query"]["feat"], position_ids=batch["query"]["feat_pos_id"], token_type_ids=batch["query"]["feat_token_id"], attention_mask=batch["query"]["feat_mask"])

        return query_output

    def VSMMA(self, batch):
        video_output = dict()

        if len(batch["vid"]["feat"].size()) == 4:
            bsz, num_video = batch["vid"]["feat"].size()[:2]
            for key in batch.keys():
                if key in ["vid", "sub"]:
                    for key_2 in batch[key]:
                        if key_2 in ["feat", "feat_mask", "feat_pos_id", "feat_token_id"]:
                            shape_list = batch[key][key_2].size()[2:]
                            batch[key][key_2] = batch[key][key_2].view((bsz * num_video,) + shape_list)


        video_output["vid"], video_output["sub"] = self.vsMMA(
            vid_features=batch["vid"]["feat"],
            vid_position_ids=batch["vid"]["feat_pos_id"],
            vid_token_type_ids=batch["vid"]["feat_token_id"],
            vid_attention_mask=batch["vid"]["feat_mask"],
            text_features=batch["sub"]["feat"],
            text_position_ids=batch["sub"]["feat_pos_id"],
            text_token_type_ids=batch["sub"]["feat_token_id"],
            text_attention_mask=batch["sub"]["feat_mask"]
        )

        return video_output


class NVLDModel(nn.Module):
    def __init__(self, config):
        super(NVLDModel, self).__init__()

        self.text_pooling = NetVLAD(feature_size=config.hidden_size,cluster_size=config.text_cluster)
        self.dropout = nn.Dropout(config.moe_dropout_prob)

        self.fc_lyr = nn.Linear(in_features=self.text_pooling.out_dim, out_features=2, bias=False)


    def forward(self, query_feat):
        pooled_text = self.text_pooling(query_feat)
        pooled_text = self.dropout(pooled_text)
        moe_weights = self.fc_lyr(pooled_text)
        softmax_moe_weights = F.softmax(moe_weights, dim=1)

        moe_weights_dict = dict()
        moe_weights_dict['vid'] = softmax_moe_weights[:,0]
        moe_weights_dict['sub'] = softmax_moe_weights[:,1]

        return  moe_weights_dict

class VQMMApool(nn.Module):

    def __init__(self, video_dim):
        super(VQMMApool, self).__init__()
        self.sim_weight = nn.Sequential(nn.Linear(video_dim * 3, video_dim, bias=False), nn.ReLU(), nn.Linear(video_dim, 1, bias=False))

    def forward(self, vid_feat, query_feat, vid_mask, query_mask):

        video_len = vid_feat.size()[1]
        query_len = query_feat.size()[1]

        _vid_feat = vid_feat.unsqueeze(2).repeat(1, 1, query_len, 1)
        _query_feat = query_feat.unsqueeze(1).repeat(1, video_len, 1, 1)

        # Cascaded Cross-modal Attention in VLANet
        vqmul = torch.mul(_vid_feat, _query_feat)
        vqfeat = torch.cat([_vid_feat, _query_feat, vqmul], dim=3)
        sim_matrix = self.sim_weight(vqfeat).view(-1, video_len, query_len)
        sim_matrix_mask = torch.einsum("bn,bm->bnm", vid_mask, query_mask)
        DenseAtt_V2Q = F.softmax(mask_logits(sim_matrix, sim_matrix_mask), dim=-1)
        V2Q = torch.bmm(DenseAtt_V2Q, query_feat)

        DenseAtt_Q2V = F.softmax(torch.max(mask_logits(sim_matrix, sim_matrix_mask), 2)[0], dim=-1)
        DenseAtt_Q2V = DenseAtt_Q2V.unsqueeze(1)
        Q2V = torch.bmm(DenseAtt_Q2V, vid_feat)
        Q2V = Q2V.repeat(1, video_len, 1)
        Q2V2Q = torch.mul(vid_feat, Q2V)

        out = torch.cat([vid_feat, V2Q, Q2V2Q], dim=2)
        return out

