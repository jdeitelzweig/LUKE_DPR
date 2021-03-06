#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "question_entity_ids",
        "question_entity_position_ids",
        "question_entity_segments",
        "context_ids",
        "ctx_segments",
        "ctx_entity_ids",
        "ctx_entity_position_ids",
        "ctx_entity_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        ent_ids: T,
        ent_segments: T,
        ent_attn_mask: T,
        ent_position_ids: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
        entity_list_representation=True,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        random_entity_output = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states, random_entity_output = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        ent_ids,
                        ent_segments,
                        ent_attn_mask,
                        ent_position_ids,
                        representation_token_pos=representation_token_pos,
                        entity_list_representation=entity_list_representation,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states, random_entity_output = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    ent_ids,
                    ent_segments,
                    ent_attn_mask,
                    ent_position_ids,
                    representation_token_pos=representation_token_pos,
                    entity_list_representation=entity_list_representation,
                )

        return sequence_output, pooled_output, hidden_states, random_entity_output

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        question_entity_ids: T,
        question_entity_segments: T,
        question_entity_attn_mask: T,
        question_entity_position_ids: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        context_entity_ids: T,
        ctx_entity_segments: T,
        ctx_entity_attn_mask: T,
        ctx_entity_position_ids: T,
        encoder_type: str = None,
        representation_token_pos=0,
        entity_list_representation=True,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden, q_ent_out = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            question_entity_ids,
            question_entity_segments,
            question_entity_attn_mask,
            question_entity_position_ids,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            entity_list_representation=entity_list_representation,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden, ctx_ent_out = self.get_representation(
            ctx_encoder, 
            context_ids, 
            ctx_segments, 
            ctx_attn_mask,
            context_entity_ids,
            ctx_entity_segments,
            ctx_entity_attn_mask,
            ctx_entity_position_ids,
            self.fix_ctx_encoder,
            entity_list_representation=entity_list_representation,
        )

        return q_pooled_out, ctx_pooled_out, q_ent_out, ctx_ent_out

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        question_entity_tensors = []
        question_entity_position_ids = []
        ctx_tensors = []
        ctx_entity_tensors = []
        ctxs_entity_position_ids = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = []
            sample_ctxs_entity_tensors = []
            sample_ctxs_entity_position_ids = []

            for ctx in all_ctxs:
                ctx_token_ids, ctx_entity_ids, ctx_entity_position_ids = tensorizer.text_to_tensor(
                    ctx.text, 
                    title=ctx.title if (insert_title and ctx.title) else None,
                    entities=ctx.entities,
                    entity_spans=ctx.entity_spans
                )

                sample_ctxs_tensors.append(ctx_token_ids)
                sample_ctxs_entity_tensors.append(ctx_entity_ids)
                sample_ctxs_entity_position_ids.append(ctx_entity_position_ids)

            ctx_tensors.extend(sample_ctxs_tensors)
            ctx_entity_tensors.extend(sample_ctxs_entity_tensors)
            ctxs_entity_position_ids.extend(sample_ctxs_entity_position_ids)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question.text, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    token_ids, entity_ids, entity_position_ids = tensorizer.text_to_tensor(" ".join([query_token, question.text]), entities=question.entities, entity_spans=question.entity_spans)
                    question_tensors.append(token_ids)
                    question_entity_tensors.append(entity_ids)
                    question_entity_position_ids.append(entity_position_ids)
            else:
                token_ids, entity_ids, entity_position_ids = tensorizer.text_to_tensor(question.text, entities=question.entities, entity_spans=question.entity_spans)
                question_tensors.append(token_ids)
                question_entity_tensors.append(entity_ids)
                question_entity_position_ids.append(entity_position_ids)

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0) 
        ctxs_entity_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_entity_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
        questions_entity_tensor = torch.cat([q.view(1, -1) for q in question_entity_tensors], dim=0)

        try:
            get_max_mention_length = tensorizer.get_max_mention_length()
            question_entity_position_ids = torch.cat([q.view(1, -1, get_max_mention_length) for q in question_entity_position_ids], dim=0)
            ctxs_entity_position_ids = torch.cat([ctx.view(1, -1, get_max_mention_length) for ctx in ctxs_entity_position_ids], dim=0)
        except AttributeError:
            question_entity_position_ids = torch.cat([ctx for ctx in question_entity_position_ids], dim=0)
            ctxs_entity_position_ids = torch.cat([ctx for ctx in ctxs_entity_position_ids], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        question_entity_segments = torch.zeros_like(questions_entity_tensor)
        ctx_entity_segments = torch.zeros_like(ctxs_entity_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            questions_entity_tensor,
            question_entity_position_ids,
            question_entity_segments,
            ctxs_tensor,
            ctx_segments,
            ctxs_entity_tensor,
            ctxs_entity_position_ids,
            ctx_entity_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        q_ent_vectors: T,
        ctx_ent_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


class BiEncoderEntLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        q_ent_vectors: T,
        ctx_ent_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors, q_ent_vectors, ctx_ent_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T, q_ent_vector, ctx_ent_vectors) -> T:
        f = BiEncoderEntLoss.get_similarity_function()
        return f(q_vector, ctx_vectors) + f(q_ent_vector, ctx_ent_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
