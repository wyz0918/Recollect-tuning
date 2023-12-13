
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import BertTokenizer,BertPreTrainedModel,BertModel,AdamW,get_linear_schedule_with_warmup,BertConfig,RobertaTokenizer,\
    AlbertConfig, AlbertTokenizer

from models import BertForSpanMarkerNER2, AlbertForSpanMarkerNER

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import math
from tqdm import tqdm
import re
import timeit
from collections import defaultdict
# import nltk

# nltk.download('averaged_perceptron_tagger')
logger = logging.getLogger(__name__)


class ACEDatasetNER(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        print('file_path:',file_path)
        assert os.path.isfile(file_path)

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        if args.data_dir.find('ace') != -1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
        elif args.data_dir.find('scierc') != -1:
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']
        else:
            self.ner_label_list = ['NIL', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
                                   'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME',
                                   'WORK_OF_ART']

        self.max_pair_length = args.max_pair_length

        self.max_entity_length = args.max_pair_length * 2






        self.initialize()

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token

    def initialize(self):
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                    isinstance(tokenizer,RobertaTokenizer)
                    and (text[0] != "'")
                    and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        maxL = 0
        maxR = 0
        maxEL = 0
        maxSuTokenL = 0

        for l_idx, line in enumerate(f):
            # print('l_idex',l_idx,'line:',line[3680:3689])
            try:
                data = json.loads(line, strict=False)
            except json.JSONDecodeError as e:
                print("Json解码错误:", e)
            # if len(self.data) > 5:
            #     break

            # if self.args.output_dir.find('test') != -1:
            #     if len(self.data) > 5:
            #         break

            sentences = data['sentences']
            # print('sentences',sentences)

            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    sentences[i][j] = self.get_original_token(sentences[i][j])

            ners = data['ner']
            relations = data['relations']
            doc_key = data["doc_key"]
            sentence_boundaries = [0]
            words = []
            L = 0
            words_L = []
            words_L.append(0)

            for i in range(len(sentences)):
                # print(len(sentences[i]))
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(len(tokens), maxL)
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

            gold_entity = defaultdict(list)

            overflag = 0
            over_lap = []
            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]
                s_e = []

                for start, end, label in sentence_ners:
                    if s_e:
                        for s_e_i in s_e:
                            if start == s_e_i[0] or end == s_e_i[1] :
                                # overflag = 1
                                over_lap.append((start, end))
                                over_lap.append(s_e_i)

                        s_e.append((start, end))

                    else:
                        s_e.append((start, end))

                # if overflag == 1:
                #     break
            # print(over_lap)
            # if overflag == 1:
            #     print(doc_key)
                # break

            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]
                overflag = 0
                self.tot_recall += len(sentence_ners)
                entity_labels = {}
                s = []
                e = []


                # for start, end, label in sentence_ners:
                #     if start in s:
                #         overflag = 1
                #         break
                #     else:
                #         s.append(start)
                #         e.append(end)
                #
                #     if end in e:
                #         overflag = 1
                #         break
                #     else:
                #         s.append(start)
                #         e.append(end)
                #
                # if overflag == 1:
                #     break

                for start, end, label in sentence_ners:
                    entity_labels[(token2subword[start], token2subword[end + 1])] = ner_label_map[label]
                    gold_entity[label].append([start, end,start, end])
                    self.ner_golden_labels.add(((l_idx, n), (start, end), label))


                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n: n + 2]

                # maxSuTokenL = max( maxSuTokenL,max_num_subwords)

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start


                maxSuTokenL = max( maxSuTokenL,sentence_length)


                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if half_context_length<0:
                    print('Yes!!!???')
                    break




                if left_length < right_length:

                    # if half_context_length < 0:
                    #     half_context_length = left_length
                    left_context_length = min(left_length, half_context_length)

                    tmp = max_num_subwords - left_context_length - sentence_length
                    if tmp<0:
                        tmp = 0
                    right_context_length = min(right_length, tmp)
                else:
                    # if half_context_length < 0:
                    #     half_context_length = right_length


                    right_context_length = min(right_length, half_context_length)
                    tmp = max_num_subwords - right_context_length - sentence_length
                    if tmp < 0:
                        tmp = 0
                    left_context_length = min(left_length, tmp)
                if self.args.output_dir.find('ctx0') != -1:
                    left_context_length = right_context_length = 0  # for debug

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset: doc_sent_end + right_context_length]


                if len(target_tokens)>max_num_subwords:
                    print(len(target_tokens))
                # assert (len(target_tokens) <= max_num_subwords)


                # tw = pos_tag(words)
                # nounwords = []
                # for i_w in tw:
                #     if i_w[1] not in ('CD','JJ','JJR','JJS','RB','RBS','RBR','VB','TO','UH','VBD','VBG','VBN','VBG','VBP','VBZ') and i_w[0] not in nounwords:
                #         nounwords.append(i_w[0])

                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]



                entity_infos = []
                entity_infos2 = []

                for entity_start in range(left_context_length, left_context_length + sentence_length):
                    doc_entity_start = entity_start + doc_offset
                    if doc_entity_start not in subword_start_positions:
                        continue

                    for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                        doc_entity_end = entity_end + doc_offset
                        if doc_entity_end not in subword_start_positions:
                            continue

                        if subword2token[doc_entity_end - 1] - subword2token[
                            doc_entity_start] + 1 > self.args.max_mention_ori_length:
                            continue

                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)

                        entity_labels.pop((doc_entity_start, doc_entity_end), None)

                        # entity_infos.append(((entity_start + 1, entity_end), label,
                        #                      (subword2token[doc_entity_start], subword2token[doc_entity_end - 1]),(tw_start,tw_end)))

                        entity_infos.append(((entity_start + 1, entity_end), label,
                                             (subword2token[doc_entity_start], subword2token[doc_entity_end - 1]),
                                             ))


                        # if  entity_start + 1<0 :
                        #     print("NNNNNNNNNOOOOOOOO!!!!!!")
                        # if  entity_end<0:
                        #     print("NNNNNNNNNOOOOOOOO!!!!!!")



                # if len(entity_labels):
                #     print ((entity_labels))
                # assert(len(entity_labels)==0)

                dL = self.max_pair_length
                # maxR = max(maxR, len(entity_infos))
                # for i in range(0, len(entity_infos), dL):
                #     examples = entity_infos[i : i + dL]
                #     item = {
                #         'sentence': target_tokens,
                #         'examples': examples,
                #         'example_index': (l_idx, n),
                #         'example_L': len(entity_infos)
                #     }

                #     self.data.append(item)
                # replace = []

                for i in range(0, len(entity_infos), dL):
                    examples = entity_infos[i: i + dL]
                    item = {
                        'sentence': target_tokens,
                        'examples': examples,
                        'example_index': (l_idx, n),
                        'example_L': len(entity_infos)
                    }
                    self.data.append(item)


            # words2 = words
            # sentences2 = []
            # ners2 = ners
            # relations2 = relations
            # replace_map = {}
            # sentence_offset = [0]
            # sen_off = 0
            #
            # used_value = []
            #
            # for key, value in gold_entity.items():
            #     value2 = np.array(value)
            #     np.random.shuffle(value2)
                # output_w = open('new_train.json', 'a+')
                # output_w.write(json.dumps(value) + '\n'+json.dumps(value2) + '\n')

                # if len(value) > 1:
                #     for value_i in value:
                #         if (value_i[2],value_i[3]) in over_lap:
                #             # print((value_i[2],value_i[3]))
                #             continue
                #         value2 = np.array(value)
                #         np.random.shuffle(value2)
                #         value2 = value2.tolist()
                #         # value2 = value
                #         for value_j in value2:
                #
                #             if (value_j[2], value_j[3]) in used_value or value_i == value_j:
                #                 # print((value_i[2],value_i[3]))
                #                 continue
                #             # if words2[value_j[0]] != words2[value_i[0]]:
                #                 # print(value_j,value_i)
                #                 # tmp_value = None
                #                 # tmp_off = 0
                #
                #             words2 = words2[:value_i[0]] + words2[value_j[0]:value_j[1]+1] + words2[value_i[1]+ 1:]
                #
                #             off_2 = value_j[1] - value_j[0]
                #             off_1 = value_i[1] - value_i[0]
                #
                #             off = off_2 - off_1
                #
                #             # start = value_i[0]
                #             value_i[1] = value_i[1]+ off
                #
                #             # value2 = value
                #             # sen_off += off
                #
                #
                #
                #             for k, v in gold_entity.items():
                #                 for value_k in v:
                #                     if value_k[0] > value_i[0]:
                #                         value_k[0] += off
                #                         value_k[1] += off
                #
                #
                #
                #             if value_j[0] > value_i[0]:
                #                 value_j[0] += off
                #                 value_j[1] += off
                #
                #             used_value.append((value_j[2], value_j[3]))
                #
                #             break
                                # if value2==value:
                                #     print("yes")
                                # else:
                                #     print("no")

                                # for value_ii in value2:
                                #     if value_ii[0] > value_i[0]:
                                #         value_ii[0] += off
                                #         value_ii[1] += off

                        # value2 = np.array(value)
                        # np.random.shuffle(value2)
                        # value2 = value2.tolist()

                        # if tmp_value:
                        #     value2.remove(tmp_value)



                        # replace_map[(value_i[0],value_i[1])] = (value_j[0],value_j[1])
                        # if off!=0:
                        #     print('yes')
                        #
                        #
                        # if value_j not in value2:
                        #     print(value2,value_j)


            # for key, value in gold_entity.items():
            #     for v_i in value:
            #         replace_map[(v_i[2],v_i[3])] = (v_i[0],v_i[1])
            #
            # for ner in ners2:
            #     for ner_i in ner:
            #
            #         new_ner = replace_map[(ner_i[0],ner_i[1])]
            #
            #         if (ner_i[0],ner_i[1]) != new_ner:
            #             ner_i[0] = new_ner[0]
            #             ner_i[1] = new_ner[1]
            #
            # for rel in relations2:
            #     for rel_i in rel:
            #         new_rel_i = replace_map[(rel_i[0], rel_i[1])]
            #         new_rel_j = replace_map[(rel_i[2], rel_i[3])]
            #
            #         if (rel_i[0], rel_i[1]) != new_rel_i:
            #             rel_i[0] = new_rel_i[0]
            #             rel_i[1] = new_rel_i[1]
            #
            #         if (rel_i[2], rel_i[3]) != new_rel_j:
            #             rel_i[2] = new_rel_j[0]
            #             rel_i[3] = new_rel_j[1]

            # print('replace_map',replace_map)






            # for i in sentence_boundaries:
            #     # print(len(sentences[i]))
            #     L += len(sentences[i])
            #     sentence_boundaries.append(L)
            #     words += sentences[i]

            # pre_pos = 0
            # for i_index, i_word in enumerate(words2):
            #     if i_word=='.':
            #         sentences2.append(words2[pre_pos:i_index+1])
            #         pre_pos = i_index+1





            # for n in range(len(sentence_boundaries) - 1):
            #
            #     if n != len(sentence_boundaries) - 1:
            #         # print('sentence_boundaries[n]',sentence_offset[n])
            #         s,e = sentence_boundaries[n]+sentence_offset[n], sentence_boundaries[n+1]+sentence_offset[n+1]
            #     else:
            #         sentences2.append(words2[sentence_boundaries[n]:])
                # for i_l in range(len(words_L)):
                #
                #     if i_l!= len(words_L)-1:
                #
                #         # print('words2[words_L[i_l]: words_L[i_l + 1]]',words2[words_L[i_l]: words_L[i_l + 1]])
                #         sentences2.append(words2[words_L[i_l]: words_L[i_l + 1]])
                #     else:
                #         sentences2.append(words2[words_L[i_l]:])

            #
            # data2 = data
            # data2['sentences'] = sentences2
            #
            # data2['ner'] = ners2
            # data2['relations'] = relations2
            # output_w = open('new_train.json', 'a+')
            # output_w.write(json.dumps(data2) + '\n')
            # break















        # print(self.data)

        # exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)


        input_ids += [0] * (self.max_seq_length - len(input_ids))
        position_plus_pad = int(self.model_type.find('roberta') != -1) * 2

        # l_m = 'start'
        # r_m = 'end'
        # l_m = self.tokenizer._convert_token_to_id(l_m)
        # r_m = self.tokenizer._convert_token_to_id(r_m)


        if self.model_type not in ['bertspan', 'robertaspan', 'albertspan']:

            if self.model_type.startswith('albert'):
                input_ids = input_ids + [30000] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [30001] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
            elif self.model_type.startswith('roberta'):
                input_ids = input_ids + [50261] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [50262] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
            else:
                input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                # input_ids = input_ids + [3] * (len(entry['examples'])) + [0] * (
                #         self.max_pair_length - len(entry['examples']))
                # input_ids = input_ids + [4] * (len(entry['examples'])) + [0] * (
                #         self.max_pair_length - len(entry['examples']))



        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [
                0] * self.max_entity_length

        labels = []
        mentions = []
        mention_pos = []
        boundary_width = []
        num_pair = self.max_pair_length
        position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [
            0] * self.max_entity_length
        input_mask = [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_entity_length) * 1
        id_start = []
        id_end = []
        for x_idx, x in enumerate(entry['examples']):
            m1 = x[0]
            label = x[1]
            # tw = x[3]
            mentions.append(x[2])
            mention_pos.append((m1[0], m1[1]))
            # boundary_width.append(m1[1]-m1[0]+1)
            labels.append(label)

            # tw_start = self.tokenizer.convert_tokens_to_ids(tw[0])
            # tw_end = self.tokenizer.convert_tokens_to_ids(tw[1])


            w1 = x_idx
            w2 = w1 + num_pair
            w3 = w2 + num_pair
            w4 = w3 + num_pair


            w1 += self.max_seq_length
            w2 += self.max_seq_length

            w3 += self.max_seq_length
            w4 += self.max_seq_length

            position_ids[w1] = m1[0]
            position_ids[w2] = m1[1]


            # print('id_start:',[input_ids[m1[0]]])

            for xx in [w1]:
                input_mask[xx] = 2

            for xx in [w2]:
                input_mask[xx] = 3


                # for yy in [w1, w2]:
                #     attention_mask[xx, yy] = 1
                # attention_mask[xx, :L] = 1


            # for xx in [w3, w4]:
            #     full_attention_mask[xx] = 1
            #     for yy in [w3, w4]:
            #         attention_mask[xx, yy] = 1
            #     attention_mask[xx, :L] = 1

        # input_ids = input_ids + id_start + [0] * (
        #                 self.max_pair_length - len(entry['examples']))+ id_end +[0] * (
        #                 self.max_pair_length - len(entry['examples']))

        attention_mask = []
        for _, from_mask in enumerate(input_mask):
            attention_mask_i = []
            for to_mask in input_mask:
                if to_mask <= 1:
                    attention_mask_i.append(to_mask)
                elif from_mask == to_mask and from_mask > 0:
                    attention_mask_i.append(1)
                else:
                    attention_mask_i.append(0)
            attention_mask.append(attention_mask_i)

        # print('input_ids_shape:',input_ids.shape)
        labels += [-1] * (num_pair - len(labels))
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))
        boundary_width += [0] * (num_pair - len(boundary_width))

        item = [torch.tensor(input_ids),
                torch.tensor(attention_mask),
                torch.tensor(position_ids),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(mention_pos),
                # torch.tensor(boundary_width),
                # torch.tensor(full_attention_mask)
                ]
        # print(torch.tensor(mention_pos))
        # print(torch.tensor(mention_pos).shape)

        if self.evaluate:
            item.append(entry['example_index'])
            item.append(mentions)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        # print('fields:',fields)

        num_metadata_fields = 2
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)







def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter(
            "logs/" + args.data_dir[max(args.data_dir.rfind('/'), 0):] + "_ner_logs/" + args.output_dir[
                                                                                        args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDatasetNER(tokenizer=tokenizer, args=args)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=2 * int(args.output_dir.find('test') == -1))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps == -1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:





        # set up GPU
        model = torch.nn.DataParallel(model)







    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1

    for _ in train_iterator:
        # if _ > 0 and (args.shuffle or args.group_edge or args.group_sort):
        #     train_dataset.initialize()
        #     if args.group_edge:
        #         train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        #         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      'labels': batch[3],
                      # 'boundary_width':batch[5],
                      # 'full_attention_mask':batch[5],
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]


            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']
                        tb_writer.add_scalar('f1', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print('Best F1', best_f1)
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset = ACEDatasetNER(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)
    ner_golden_labels = set(eval_dataset.ner_golden_labels)
    ner_tot_recall = eval_dataset.tot_recall

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=ACEDatasetNER.collate_fn,
                                 num_workers=4 * int(args.output_dir.find('test') == -1))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    scores = defaultdict(dict)
    predict_ners = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-2]
        batch_m2s = batch[-1]

        batch = tuple(t.to(args.device) for t in batch[:-2])

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      #   'labels':         batch[3]
                      # 'full_attention_mask' : batch[5],
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]

            outputs = model(**inputs)

            ner_logits = outputs[0]
            ner_logits = torch.nn.functional.softmax(ner_logits, dim=-1)
            ner_values, ner_preds = torch.max(ner_logits, dim=-1)

            for i in range(len(indexs)):
                index = indexs[i]
                m2s = batch_m2s[i]
                for j in range(len(m2s)):
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]
                    if ner_label != 'NIL':
                        scores[(index[0], index[1])][(obj[0], obj[1])] = (float(ner_values[i, j]), ner_label)

    cor = 0
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0

    for example_index, pair_dict in scores.items():

        sentence_results = []
        for k1, (v2_score, v2_ner_label) in pair_dict.items():
            if v2_ner_label != 'NIL':
                sentence_results.append((v2_score, k1, v2_ner_label))

        sentence_results.sort(key=lambda x: -x[0])
        no_overlap = []

        def is_overlap(m1, m2):
            if m2[0] <= m1[0] and m1[0] <= m2[1]:
                return True
            if m1[0] <= m2[0] and m2[0] <= m1[1]:
                return True
            return False

        for item in sentence_results:
            m2 = item[1]
            overlap = False
            for x in no_overlap:
                _m2 = x[1]
                if (is_overlap(m2, _m2)):
                    if args.data_dir.find('ontonotes') != -1:
                        overlap = True
                        break
                    else:

                        if item[2] == x[2]:
                            overlap = True
                            break

            if not overlap:
                no_overlap.append(item)

            pred_ner_label = item[2]
            tot_pred_tot += 1
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor_tot += 1

        for item in no_overlap:
            m2 = item[1]
            pred_ner_label = item[2]
            tot_pred += 1
            if args.output_results:
                predict_ners[example_index].append((m2[0], m2[1], pred_ner_label))
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor += 1

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime, len(eval_dataset) / evalTime)

    precision_score = p = cor / tot_pred if tot_pred > 0 else 0
    recall_score = r = cor / ner_tot_recall
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0
    r = cor_tot / ner_tot_recall
    f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    results = {'f1': f1, 'f1_overlap': f1_tot, 'precision': precision_score, 'recall': recall_score}

    logger.info("Result: %s", json.dumps(results))

    if args.output_results and (do_test or not args.do_train):
        f = open(eval_dataset.file_path)
        if do_test:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_test.json'), 'w')
        else:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_dev.json'), 'w')
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            num_sents = len(data['sentences'])
            predicted_ner = []
            for n in range(num_sents):
                item = predict_ners.get((l_idx, n), [])
                item.sort()
                predicted_ner.append(item)

            data['predicted_ner'] = predicted_ner
            output_w.write(json.dumps(data) + '\n')

    return results


def main():
    # os.environ['MASTER_ADDR'] = '10.10.10.219'
    # os.environ['MASTER_PORT'] = '20066'
    # os.environ["NCCL_DEBUG"] = "INFO"
    parser = argparse.ArgumentParser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"



    ## Required parameters
    parser.add_argument("--data_dir", default='datasets/scierc', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bertspanmarker', type=str,
                        )
    parser.add_argument("--model_name_or_path", default='bert_models/scibert_scivocab_uncased', type=str,
                        )
    parser.add_argument("--output_dir", default='sicner_bert4/sci-bert', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train",action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")

    parser.add_argument("--evaluate_during_training", default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", default=True,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default='ace05ner_models/PL-Marker-ace05-bert',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', default=False,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # parser.add_argument('--fp16', default=False,
    #                     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16', default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file", default="train_data.json", type=str)
    parser.add_argument("--dev_file", default="test_data.json", type=str)
    parser.add_argument("--test_file", default="test_data.json", type=str)

    parser.add_argument('--alpha', type=float, default=1, help="")

    parser.add_argument('--max_pair_length', type=int, default=40, help="")

    parser.add_argument('--max_mention_ori_length', type=int, default=6, help="")
    parser.add_argument('--lminit', default=True)
    parser.add_argument('--norm_emb', action='store_true')
    parser.add_argument('--output_results', default=True)
    parser.add_argument('--onedropout', default=True)
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--use_full_layer', type=int, default=-1, help="")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--group_edge', action='store_true')
    parser.add_argument('--group_axis', type=int, default=-1, help="")
    parser.add_argument('--group_sort', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test') == -1:
        create_exp_dir(args.output_dir,
                       scripts_to_save=['run_ner2.py', 'models.py',])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',rank=args.local_rank)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if args.data_dir.find('ace') != -1:
        num_labels = 8
    elif args.data_dir.find('scierc') != -1:
        num_labels = 7
    elif args.data_dir.find('ontonotes') != -1:
        num_labels = 19
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    if args.model_type.startswith('albert'):
        config_class, model_class, tokenizer_class = AlbertConfig, AlbertForSpanMarkerNER, AlbertTokenizer
    else:
        config_class, model_class, tokenizer_class = BertConfig, BertForSpanMarkerNER2, BertTokenizer



    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer = args.use_full_layer

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.model_type.startswith('albert'):
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit:
        if args.model_type.find('roberta') == -1:
            entity_id = tokenizer.encode('entity', add_special_tokens=False)
            assert (len(entity_id) == 1)
            entity_id = entity_id[0]
            mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
            assert (len(mask_id) == 1)
            mask_id = mask_id[0]
        else:
            entity_id = 10014
            mask_id = 50264

        logger.info('entity_id: %d', entity_id)
        logger.info('mask_id: %d', mask_id)

        if args.model_type.startswith('albert'):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])
            word_embeddings[30001].copy_(word_embeddings[entity_id])
        elif args.model_type.startswith('roberta'):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])  # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id])
        else:
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])
            word_embeddings[2].copy_(word_embeddings[entity_id])  # entity

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results['f1']
            if f1 > best_f1:
                best_f1 = f1
                print('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate on test set")

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))


if __name__ == "__main__":
    main()
