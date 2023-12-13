import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


import  numpy as np

from transformers import BertTokenizer,BertPreTrainedModel,BertModel,AlbertPreTrainedModel,AlbertModel

import os,json,logging


class AlbertForSpanMarkerNER6(AlbertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)
        # self.ner_classifier_total = nn.Sequential(
        #     FeedForward(
        #         # input_dim = config.hidden_size*4+width_embedding_dim,
        #         input_dim=config.hidden_size * 4 +width_embedding_dim,
        #         num_layers=2,
        #         hidden_dims=head_hidden_dim,
        #         activations=torch.nn.ReLU(),
        #         dropout=0.2,
        #     ),
        #
        #     nn.Linear(head_hidden_dim, self.num_labels)
        #
        # )

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,

            mentions=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # boundary_width=None,
            # return_dict = None,
    ):
        # print('input_ids:',input_ids.shape)
        # print('full_attention_mask:',full_attention_mask.shape)
        # print('attention_mask:',attention_mask[:,:,0].shape)
        #
        # print('position_ids:',position_ids.shape)
        # print('inputs_embeds:',inputs_embeds.shape)



        outputs = self.albert(
            input_ids,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            # return_dict = return_dict,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 6

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)

        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4 + ner_prediction_scores5 + ner_prediction_scores6

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class AlbertForSpanMarkerNER4(AlbertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(0.2)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)
        # self.ner_classifier_total = nn.Sequential(
        #     FeedForward(
        #         # input_dim = config.hidden_size*4+width_embedding_dim,
        #         input_dim=config.hidden_size * 4 +width_embedding_dim,
        #         num_layers=2,
        #         hidden_dims=head_hidden_dim,
        #         activations=torch.nn.ReLU(),
        #         dropout=0.2,
        #     ),
        #
        #     nn.Linear(head_hidden_dim, self.num_labels)
        #
        # )

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,

            mentions=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # boundary_width=None,
            # return_dict = None,
    ):
        # print('input_ids:',input_ids.shape)
        # print('full_attention_mask:',full_attention_mask.shape)
        # print('attention_mask:',attention_mask[:,:,0].shape)
        #
        # print('position_ids:',position_ids.shape)
        # print('inputs_embeds:',inputs_embeds.shape)



        outputs = self.albert(
            input_ids,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            # return_dict = return_dict,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]

        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)

        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs



class AlbertForSpanMarkerNER(AlbertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(0.2)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)
        # self.ner_classifier_total = nn.Sequential(
        #     FeedForward(
        #         # input_dim = config.hidden_size*4+width_embedding_dim,
        #         input_dim=config.hidden_size * 4 +width_embedding_dim,
        #         num_layers=2,
        #         hidden_dims=head_hidden_dim,
        #         activations=torch.nn.ReLU(),
        #         dropout=0.2,
        #     ),
        #
        #     nn.Linear(head_hidden_dim, self.num_labels)
        #
        # )

        self.ner_classifier1 = nn.Linear(config.hidden_size*3, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*3, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,

            mentions=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # boundary_width=None,
            # return_dict = None,
    ):
        # print('input_ids:',input_ids.shape)
        # print('full_attention_mask:',full_attention_mask.shape)
        # print('attention_mask:',attention_mask[:,:,0].shape)
        #
        # print('position_ids:',position_ids.shape)
        # print('inputs_embeds:',inputs_embeds.shape)



        outputs = self.albert(
            input_ids,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            # return_dict = return_dict,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]


        feature_vector1 = torch.cat([m1_start_states,  e1_hidden_states, e3_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_end_states, e2_hidden_states, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class AlbertForSpanMarkerNER8(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)
        # self.ner_classifier_total = nn.Sequential(
        #     FeedForward(
        #         # input_dim = config.hidden_size*4+width_embedding_dim,
        #         input_dim=config.hidden_size * 4 +width_embedding_dim,
        #         num_layers=2,
        #         hidden_dims=head_hidden_dim,
        #         activations=torch.nn.ReLU(),
        #         dropout=0.2,
        #     ),
        #
        #     nn.Linear(head_hidden_dim, self.num_labels)
        #
        # )

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,

            mentions=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # boundary_width=None,
            # return_dict = None,
    ):
        # print('input_ids:',input_ids.shape)
        # print('full_attention_mask:',full_attention_mask.shape)
        # print('attention_mask:',attention_mask[:,:,0].shape)
        #
        # print('position_ids:',position_ids.shape)
        # print('inputs_embeds:',inputs_embeds.shape)



        outputs = self.albert(
            input_ids,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            # return_dict = return_dict,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 8

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:]

        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)

        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m1_states7 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m1_states8 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)
        feature_vector7 = torch.cat([m1_states7, e7_hidden_states], dim=2)
        feature_vector8 = torch.cat([m1_states8, e8_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)
            feature_vector7 = self.dropout(feature_vector7)
            feature_vector8 = self.dropout(feature_vector8)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4+ ner_prediction_scores5 + ner_prediction_scores6+ ner_prediction_scores7 + ner_prediction_scores8

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class AlbertForBinarySpanMarkerNER(AlbertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(0.2)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)
        # self.ner_classifier_total = nn.Sequential(
        #     FeedForward(
        #         # input_dim = config.hidden_size*4+width_embedding_dim,
        #         input_dim=config.hidden_size * 4 +width_embedding_dim,
        #         num_layers=2,
        #         hidden_dims=head_hidden_dim,
        #         activations=torch.nn.ReLU(),
        #         dropout=0.2,
        #     ),
        #
        #     nn.Linear(head_hidden_dim, self.num_labels)
        #
        # )

        self.ner_classifier1 = nn.Linear(config.hidden_size*3, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*3, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,

            mentions=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # boundary_width=None,
            # return_dict = None,
    ):
        # print('input_ids:',input_ids.shape)
        # print('full_attention_mask:',full_attention_mask.shape)
        # print('attention_mask:',attention_mask[:,:,0].shape)
        #
        # print('position_ids:',position_ids.shape)
        # print('inputs_embeds:',inputs_embeds.shape)



        outputs = self.albert(
            input_ids,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            # return_dict = return_dict,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]


        feature_vector1 = torch.cat([m1_start_states,  e1_hidden_states, e3_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_end_states, e2_hidden_states, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForBinarySpanMarkerNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 1, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 1, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels_start=None,
            mention_pos_start=None,
            labels_end=None,
            mention_pos_end=None,
            # full_attention_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)
        bsz, tot_seq_len = input_ids.shape
        seq_len = self.max_seq_length

        m_states_start = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos_start[:, :]]
        m_states_end = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos_end[:, :]]


        # m_states = hidden_states[:, :seq_len]

        if not self.onedropout:
            feature_vector_start = m_states_start
            feature_vector_end = m_states_end
        else:
            feature_vector_start = self.dropout(m_states_start)
            feature_vector_end = self.dropout(m_states_end)


        ner_prediction_scores1 = self.ner_classifier1(feature_vector_start)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector_end)


        outputs = (ner_prediction_scores1,ner_prediction_scores2) + outputs[2:]  # Add hidden states and attention if they are here

        if labels_start is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner1 = CrossEntropyLoss(ignore_index=-1, reduction='sum',
                                            weight=self.alpha.to(ner_prediction_scores1))
            ner_loss1 = loss_fct_ner1(ner_prediction_scores1.view(-1, self.num_labels), labels_start.view(-1))

            loss_fct_ner2 = CrossEntropyLoss(ignore_index=-1, reduction='sum',
                                             weight=self.alpha.to(ner_prediction_scores2))
            ner_loss2 = loss_fct_ner2(ner_prediction_scores2.view(-1, self.num_labels), labels_end.view(-1))

            loss = ner_loss1+ner_loss2

            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)

            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,ner_loss1,ner_loss2 ) + outputs

        return outputs

class BertForSpanMarkerNER0(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        # ent_len = (tot_seq_len - seq_len) // 2

        # marker start+end

        # e1_end = seq_len + ent_len
        # e2_end = e1_end + ent_len
        # e3_end = e2_end + ent_len
        #
        # e1_hidden_states = hidden_states[:, seq_len:e1_end]
        # e2_hidden_states = hidden_states[:, e1_end:]


        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        # m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([m1_states1, m1_states2], dim=2)
        # feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        # feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            # feature_vector2 = self.dropout(feature_vector2)
            # feature_vector3 = self.dropout(feature_vector3)
            # feature_vector4 = self.dropout(feature_vector4)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        # ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        # ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1
        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertForSpanMarkerNER2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 2

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:]


        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        # m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        # feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        # feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            # feature_vector3 = self.dropout(feature_vector3)
            # feature_vector4 = self.dropout(feature_vector4)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        # ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForSpanMarkerNER4(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForSpanMarkerNER6(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 6

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4+ ner_prediction_scores5 + ner_prediction_scores6

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForSpanMarkerNER6_wospeedup(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*1, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*1, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier4 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = tot_seq_len - seq_len

        # marker start+end

        # e1_end = seq_len + ent_len
        # e2_end = e1_end + ent_len
        # e3_end = e2_end + ent_len
        # e4_end = e3_end + ent_len
        # e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:]
        # e2_hidden_states = hidden_states[:, e1_end:e2_end]


        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:e4_end]
        #
        # e5_hidden_states = hidden_states[:, e4_end:e5_end]
        # e6_hidden_states = hidden_states[:, e5_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        # m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        # m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([m1_states1], dim=2)
        feature_vector2 = torch.cat([m1_states2], dim=2)
        feature_vector3 = torch.cat([m1_states3, e1_hidden_states], dim=2)
        # feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        # feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        # feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            # feature_vector4 = self.dropout(feature_vector4)
            # feature_vector5 = self.dropout(feature_vector5)
            # feature_vector6 = self.dropout(feature_vector6)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        # ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        # ner_prediction_scores6 = self.ner_classifier6(feature_vector6)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForSpanMarkerNER8(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )




        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 8

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:]


        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m1_states7 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m1_states8 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)
        feature_vector7 = torch.cat([m1_states7, e7_hidden_states], dim=2)
        feature_vector8 = torch.cat([m1_states8, e8_hidden_states], dim=2)


        # feature_vector9 = torch.cat([e1_hidden_states, e2_hidden_states, e3_hidden_states, e4_hidden_states, e5_hidden_states, e6_hidden_states, e7_hidden_states, e8_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)
            feature_vector7 = self.dropout(feature_vector7)
            feature_vector8 = self.dropout(feature_vector8)
            # feature_vector9 = self.dropout(feature_vector9)



        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)

        # ner_prediction_scores9 = self.ner_classifier9(feature_vector9)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4+ner_prediction_scores5 + ner_prediction_scores6+ner_prediction_scores7 + ner_prediction_scores8



        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs

class BertForSpanMarkerNER8_random(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            # full_attention_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )

        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 8

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:]

        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)

        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m1_states7 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m1_states8 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)
        feature_vector7 = torch.cat([m1_states7, e7_hidden_states], dim=2)
        feature_vector8 = torch.cat([m1_states8, e8_hidden_states], dim=2)

        # feature_vector9 = torch.cat([e1_hidden_states, e2_hidden_states, e3_hidden_states, e4_hidden_states, e5_hidden_states, e6_hidden_states, e7_hidden_states, e8_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)
            feature_vector7 = self.dropout(feature_vector7)
            feature_vector8 = self.dropout(feature_vector8)
            # feature_vector9 = self.dropout(feature_vector9)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)

        # ner_prediction_scores9 = self.ner_classifier9(feature_vector9)

        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4 + ner_prediction_scores5 + ner_prediction_scores6 + ner_prediction_scores7 + ner_prediction_scores8

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, reduction='sum',
                                            weight=self.alpha.to(ner_prediction_scores))

            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)

            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertForSpanMarkerNER10(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)

        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.width_embedding = nn.Embedding(34,width_embedding_dim)


        self.ner_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.ner_classifier10 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.boundary_width_classifier1 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            boundary_width=None,
            # full_attention_mask=None,
    ):



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 10

        # marker start+end

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len
        e8_end = e7_end + ent_len
        e9_end = e8_end + ent_len



        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]


        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]

        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:e8_end]

        e9_hidden_states = hidden_states[:, e8_end:e9_end]
        e10_hidden_states = hidden_states[:, e9_end:]
        # print('bsz:',bsz)
        # print('tot_seq_len:',tot_seq_len)
        # print('hidden_states_len:',hidden_states.shape)
        # print('mention_pos_shape',mention_pos.shape)




        # text tokens embedding
        m1_states1 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_states2 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m1_states3 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m1_states4 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m1_states5 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m1_states6 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m1_states7 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m1_states8 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]
        m1_states9 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 8]]
        m1_states10 = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 9]]

        feature_vector1 = torch.cat([m1_states1, e1_hidden_states], dim=2)
        feature_vector2 = torch.cat([m1_states2, e2_hidden_states], dim=2)
        feature_vector3 = torch.cat([m1_states3, e3_hidden_states], dim=2)
        feature_vector4 = torch.cat([m1_states4, e4_hidden_states], dim=2)
        feature_vector5 = torch.cat([m1_states5, e5_hidden_states], dim=2)
        feature_vector6 = torch.cat([m1_states6, e6_hidden_states], dim=2)
        feature_vector7 = torch.cat([m1_states7, e7_hidden_states], dim=2)
        feature_vector8 = torch.cat([m1_states8, e8_hidden_states], dim=2)
        feature_vector9 = torch.cat([m1_states9, e9_hidden_states], dim=2)
        feature_vector10 = torch.cat([m1_states10, e10_hidden_states], dim=2)

        # feature_vector9 = torch.cat([e1_hidden_states, e2_hidden_states, e3_hidden_states, e4_hidden_states, e5_hidden_states, e6_hidden_states, e7_hidden_states, e8_hidden_states], dim=2)

        # print('boundary_width_shape:',boundary_width.shape)

        # print('boundary_width:',boundary_width)

        # span_width_embedding = self.width_embedding(boundary_width)

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states, span_width_embedding], dim=2)
        # feature_vector = torch.cat([m1_start_states, e1_hidden_states, m1_end_states,e2_hidden_states], dim=2)
        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)



        # print('fv_shape:', feature_vector.shape)

        if not self.onedropout:
            # feature_vector = self.dropout(feature_vector)
            feature_vector1 = self.dropout(feature_vector1)
            feature_vector2 = self.dropout(feature_vector2)
            feature_vector3 = self.dropout(feature_vector3)
            feature_vector4 = self.dropout(feature_vector4)
            feature_vector5 = self.dropout(feature_vector5)
            feature_vector6 = self.dropout(feature_vector6)
            feature_vector7 = self.dropout(feature_vector7)
            feature_vector8 = self.dropout(feature_vector8)
            feature_vector9 = self.dropout(feature_vector9)
            feature_vector10 = self.dropout(feature_vector10)
            # feature_vector9 = self.dropout(feature_vector9)



        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)
        ner_prediction_scores9 = self.ner_classifier7(feature_vector9)
        ner_prediction_scores10 = self.ner_classifier8(feature_vector10)



        # ffnn_hidden = []
        #
        # hidden = feature_vector
        # for layer in self.ner_classifier_total:
        #     hidden = layer(hidden)
        #     ffnn_hidden.append(hidden)
        #
        # logits = ffnn_hidden[-1]

        # ner_prediction_scores = self.ner_classifier(feature_vector)

        # print('ner_prediction_scores:',ner_prediction_scores.shape)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4+ner_prediction_scores5 + ner_prediction_scores6+ner_prediction_scores7 + ner_prediction_scores8+ner_prediction_scores9 + ner_prediction_scores10



        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1,reduction='sum')

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1,  reduction='sum', weight=self.alpha.to(ner_prediction_scores))


            # print('logits:',logits)
            # print('logits_shape:',logits.shape)
            # print('labels_shape:',labels.shape)
            # print('labels:',labels)


            # ner_loss = loss_fct_ner(logits.view(-1, self.num_labels), labels.view(-1))

            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertForACEBothOneDropoutSub_PLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2
        # ent_len = (tot_seq_len - seq_len) // 4

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:]

        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:]

        # m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        # m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        # m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        # feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        # feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        # ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        ner_prediction_scores = ner_prediction_scores1


        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        # m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        # m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        # m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label



        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub0(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        # ent_len = (tot_seq_len - seq_len) // 4
        #
        # e1_end = seq_len + ent_len
        # e2_end = e1_end + ent_len
        # e3_end = e2_end + ent_len
        #
        # e1_hidden_states = hidden_states[:, seq_len:e1_end]
        # e2_hidden_states = hidden_states[:, e1_end:e2_end]

        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        # m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([m1_o_start_states,m2_o_start_states], dim=2)
        # feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        # feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        # ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        # ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        ner_prediction_scores = ner_prediction_scores1

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        # m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        # m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        # m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label



        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2
        # ent_len = (tot_seq_len - seq_len) // 4

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:]

        # e3_hidden_states = hidden_states[:, e2_end:e3_end]
        # e4_hidden_states = hidden_states[:, e3_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        # m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        # m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]

        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        # feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        # feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        # ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        # ner_prediction_scores4 = self.ner_classifier4(feature_vector4)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2


        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        # m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        # m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label



        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores + m3_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub4(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.re_classifier_m6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.re_classifier_m7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.re_classifier_m8 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]
        #
        # e5_hidden_states = hidden_states[:, e4_end:e5_end]
        # e6_hidden_states = hidden_states[:, e5_end:]



        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        # m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        # m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        # feature_vector5 = torch.cat([e5_hidden_states, e6_hidden_states,], dim=2)
        # feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        # ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        # ner_prediction_scores6 = self.ner_classifier6(feature_vector4)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4


        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        # m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        # m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label



        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores + m3_scores  + m4_scores + m5_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub6(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m7 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 6

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]
        #
        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:]



        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        feature_vector5 = torch.cat([e5_hidden_states, m5_o_start_states], dim=2)
        feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4  + ner_prediction_scores5 + ner_prediction_scores6


        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label



        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores + m3_scores  + m4_scores + m5_scores + m6_scores + m7_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub8(BertPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m8 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m9 = nn.Linear(config.hidden_size * 2, self.num_labels)

        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 8

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]
        #
        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m7_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m8_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]


        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        feature_vector5 = torch.cat([e5_hidden_states, m5_o_start_states], dim=2)
        feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)
        feature_vector7 = torch.cat([e7_hidden_states, m7_o_start_states], dim=2)
        feature_vector8 = torch.cat([e8_hidden_states, m8_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)


        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4  + ner_prediction_scores5 + ner_prediction_scores6 + ner_prediction_scores7 + ner_prediction_scores8


        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)



        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label
        m8_scores = self.re_classifier_m8(feature_vector7)  # bsz, ent_len, num_label
        m9_scores = self.re_classifier_m9(feature_vector8)


        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores + m3_scores  + m4_scores + m5_scores + m6_scores + m7_scores + m8_scores + m9_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class BertForACEBothOneDropoutSub10(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier10 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)


        self.re_classifier_sub1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub4 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m8 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m9 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m10 = nn.Linear(config.hidden_size * 2, self.num_labels)
        # self.re_classifier1 = nn.Linear(config.hidden_size*2, self.num_labels)
        # self.ner_classifier2 = nn.Linear(config.hidden_size*2, self.num_labels)


        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_mask_positions=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )


        # print('mention_pos:',mention_pos)


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 10

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len
        e8_end = e7_end + ent_len
        e9_end = e8_end + ent_len


        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]
        #
        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:e8_end]

        e9_hidden_states = hidden_states[:, e8_end:e9_end]
        e10_hidden_states = hidden_states[:, e9_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m7_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m8_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]
        m9_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 8]]
        m10_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 9]]


        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        feature_vector5 = torch.cat([e5_hidden_states, m5_o_start_states], dim=2)
        feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)
        feature_vector7 = torch.cat([e7_hidden_states, m7_o_start_states], dim=2)
        feature_vector8 = torch.cat([e8_hidden_states, m8_o_start_states], dim=2)
        feature_vector9 = torch.cat([e9_hidden_states, m9_o_start_states], dim=2)
        feature_vector10 = torch.cat([e10_hidden_states, m10_o_start_states], dim=2)


        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)
        ner_prediction_scores9 = self.ner_classifier9(feature_vector9)
        ner_prediction_scores10 = self.ner_classifier10(feature_vector10)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4  + ner_prediction_scores5 + ner_prediction_scores6 + ner_prediction_scores7 + ner_prediction_scores8 + ner_prediction_scores9 + ner_prediction_scores10

        m1_start_states1 = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states1 = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_start_states2 = hidden_states[torch.arange(bsz), sub_positions[:, 2]]
        m1_end_states2 = hidden_states[torch.arange(bsz), sub_positions[:, 3]]


        m1_mask_start_states1 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 0]]
        m1_mask_end_states1 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 1]]
        m1_mask_start_states2 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 2]]
        m1_mask_end_states2 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 3]]


        m1_states1 = torch.cat([m1_mask_start_states1,m1_mask_end_states1], dim=-1)
        # m1_states2 = torch.cat([m1_mask_end_states1], dim=-1)
        # m1_states3 = torch.cat([m1_mask_start_states2], dim=-1)
        # m1_states4 = torch.cat([m1_mask_end_states2], dim=-1)


        m1_scores1 = self.re_classifier_sub1(m1_states1)  # bsz, num_label
        # m1_scores2 = self.re_classifier_sub2(m1_states2)  # bsz, num_label
        # m1_scores3 = self.re_classifier_sub3(m1_states3)  # bsz, num_label
        # m1_scores4 = self.re_classifier_sub4(m1_states4)



        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label
        m8_scores = self.re_classifier_m8(feature_vector7)  # bsz, ent_len, num_label
        m9_scores = self.re_classifier_m9(feature_vector8)
        m10_scores = self.re_classifier_m8(feature_vector9)  # bsz, ent_len, num_label
        m11_scores = self.re_classifier_m9(feature_vector10)

        re_prediction_scores = m1_scores1.unsqueeze(1)  + m2_scores + m3_scores  + m4_scores + m5_scores + m6_scores + m7_scores + m8_scores + m9_scores + m10_scores + m11_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutSubNoNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len - seq_len) // 2
        ent_len = (tot_seq_len - seq_len) // 4




        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = 0
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class AlbertForACEBothSub(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs


class AlbertForACEBothOneDropoutSub10(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier7 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier8 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier9 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier10 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_sub1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_sub4 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m6 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m7 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m8 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m9 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m10 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_mask_positions=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 10

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len
        e6_end = e5_end + ent_len
        e7_end = e6_end + ent_len
        e8_end = e7_end + ent_len
        e9_end = e8_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:e4_end]
        #
        e5_hidden_states = hidden_states[:, e4_end:e5_end]
        e6_hidden_states = hidden_states[:, e5_end:e6_end]

        e7_hidden_states = hidden_states[:, e6_end:e7_end]
        e8_hidden_states = hidden_states[:, e7_end:e8_end]

        e9_hidden_states = hidden_states[:, e8_end:e9_end]
        e10_hidden_states = hidden_states[:, e9_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]
        m7_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 6]]
        m8_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 7]]
        m9_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 8]]
        m10_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 9]]

        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        feature_vector5 = torch.cat([e5_hidden_states, m5_o_start_states], dim=2)
        feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)
        feature_vector7 = torch.cat([e7_hidden_states, m7_o_start_states], dim=2)
        feature_vector8 = torch.cat([e8_hidden_states, m8_o_start_states], dim=2)
        feature_vector9 = torch.cat([e9_hidden_states, m9_o_start_states], dim=2)
        feature_vector10 = torch.cat([e10_hidden_states, m10_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        ner_prediction_scores6 = self.ner_classifier6(feature_vector6)
        ner_prediction_scores7 = self.ner_classifier7(feature_vector7)
        ner_prediction_scores8 = self.ner_classifier8(feature_vector8)
        ner_prediction_scores9 = self.ner_classifier9(feature_vector9)
        ner_prediction_scores10 = self.ner_classifier10(feature_vector10)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4 + ner_prediction_scores5 + ner_prediction_scores6 + ner_prediction_scores7 + ner_prediction_scores8 + ner_prediction_scores9 + ner_prediction_scores10

        m1_start_states1 = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states1 = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_start_states2 = hidden_states[torch.arange(bsz), sub_positions[:, 2]]
        m1_end_states2 = hidden_states[torch.arange(bsz), sub_positions[:, 3]]

        m1_mask_start_states1 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 0]]
        m1_mask_end_states1 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 1]]
        m1_mask_start_states2 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 2]]
        m1_mask_end_states2 = hidden_states[torch.arange(bsz), sub_mask_positions[:, 3]]

        m1_states1 = torch.cat([m1_mask_start_states1, m1_mask_end_states1], dim=-1)
        # m1_states2 = torch.cat([m1_mask_end_states1], dim=-1)
        # m1_states3 = torch.cat([m1_mask_start_states2], dim=-1)
        # m1_states4 = torch.cat([m1_mask_end_states2], dim=-1)

        m1_scores1 = self.re_classifier_sub1(m1_states1)  # bsz, num_label
        # m1_scores2 = self.re_classifier_sub2(m1_states2)  # bsz, num_label
        # m1_scores3 = self.re_classifier_sub3(m1_states3)  # bsz, num_label
        # m1_scores4 = self.re_classifier_sub4(m1_states4)

        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label
        m8_scores = self.re_classifier_m8(feature_vector7)  # bsz, ent_len, num_label
        m9_scores = self.re_classifier_m9(feature_vector8)
        m10_scores = self.re_classifier_m8(feature_vector9)  # bsz, ent_len, num_label
        m11_scores = self.re_classifier_m9(feature_vector10)

        re_prediction_scores = m1_scores1.unsqueeze(
            1) + m2_scores + m3_scores + m4_scores + m5_scores + m6_scores + m7_scores + m8_scores + m9_scores + m10_scores + m11_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here
        # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class AlbertForACEBothOneDropoutSub(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class AlbertForACEBothOneDropoutSub4(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier3 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier4 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier5 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        # self.ner_classifier6 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m4 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m5 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,

    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 4

        e1_end = seq_len + ent_len
        e2_end = e1_end + ent_len
        e3_end = e2_end + ent_len
        e4_end = e3_end + ent_len
        e5_end = e4_end + ent_len

        e1_hidden_states = hidden_states[:, seq_len:e1_end]
        e2_hidden_states = hidden_states[:, e1_end:e2_end]

        e3_hidden_states = hidden_states[:, e2_end:e3_end]
        e4_hidden_states = hidden_states[:, e3_end:]
        #
        # e5_hidden_states = hidden_states[:, e4_end:e5_end]
        # e6_hidden_states = hidden_states[:, e5_end:]

        m1_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]
        m3_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 2]]
        m4_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 3]]
        # m5_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 4]]
        # m6_o_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 5]]

        feature_vector1 = torch.cat([e1_hidden_states, m1_o_start_states], dim=2)
        feature_vector2 = torch.cat([e2_hidden_states, m2_o_start_states], dim=2)
        feature_vector3 = torch.cat([e3_hidden_states, m3_o_start_states], dim=2)
        feature_vector4 = torch.cat([e4_hidden_states, m4_o_start_states], dim=2)
        # feature_vector5 = torch.cat([e5_hidden_states, e6_hidden_states,], dim=2)
        # feature_vector6 = torch.cat([e6_hidden_states, m6_o_start_states], dim=2)

        ner_prediction_scores1 = self.ner_classifier1(feature_vector1)
        ner_prediction_scores2 = self.ner_classifier2(feature_vector2)
        ner_prediction_scores3 = self.ner_classifier3(feature_vector3)
        ner_prediction_scores4 = self.ner_classifier4(feature_vector4)
        # ner_prediction_scores5 = self.ner_classifier5(feature_vector5)
        # ner_prediction_scores6 = self.ner_classifier6(feature_vector4)

        ner_prediction_scores = ner_prediction_scores1 + ner_prediction_scores2 + ner_prediction_scores3 + ner_prediction_scores4

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector1)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(feature_vector2)  # bsz, ent_len, num_label
        m4_scores = self.re_classifier_m4(feature_vector3)  # bsz, ent_len, num_label
        m5_scores = self.re_classifier_m5(feature_vector4)  # bsz, ent_len, num_label
        # m6_scores = self.re_classifier_m6(feature_vector5)  # bsz, ent_len, num_label
        # m7_scores = self.re_classifier_m7(feature_vector6)  # bsz, ent_len, num_label

        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores + m3_scores + m4_scores + m5_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
                loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
                re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

                loss = re_loss + ner_loss
                outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)