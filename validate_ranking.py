"""
Ranking evaluation script for the trained DPR model.
"""

import logging
import math
import os
import json
import pdb
import random
import sys
import time
import collections
from typing import Tuple, List
from tqdm import tqdm
import platform

if platform.platform().startswith("Windows"):
    path = os.path.abspath(__file__)
    sys.path.append('\\'.join(path.split('\\')[:-3]))
else:
    path = os.path.abspath(__file__)
    sys.path.append('/'.join(path.split('/')[:-3]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import normalize_question
from dpr.models.biencoder import dot_product_scores
from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.utils.conf_utils import BiencoderDatasetsCfg
from dpr.utils.data_utils import (
    ShardedDataIterator,
    Tensorizer,
    MultiSetDataIterator,
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)

# torch.set_printoptions(threshold=2500)
logger = logging.getLogger()
setup_logger(logger)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
POS_IDX = 0  # The position of Q1's positive passage in each instance's context list


def mean(array: List):
    return sum(array) / len(array)


def calculate_MR(rank_results: List[torch.Tensor], pos_idx: int):
    """
    Calculate Mean Rank and Mean Reciprocal Rank of the positive passage in each ranking results.
    The index of the positive passage is given by `pos_idx`.
    """
    all_rankings = []
    all_reciprocal_rankings = []

    for indices in rank_results:
        ranking = (indices == pos_idx).nonzero().item() + 1  # ranking index starts from 1, but tensor index starts from 0
        all_rankings.append(ranking)
        all_reciprocal_rankings.append(1 / ranking)

    MR = mean(all_rankings)
    MRR = mean(all_reciprocal_rankings)
    return MR, MRR


def rank_passages(dataset, rank_results, rank_scores):
    assert len(dataset) == len(rank_results)
    verbose_results = []

    for instance, rank_idxs, scores in zip(dataset, rank_results, rank_scores):
        question = instance["question"]
        all_ctxs = instance["all_ctxs"]

        # Re-order all passages according to the ranking indices given by the model
        ranked_ctxs = []
        j = 0
        for i in rank_idxs:
            ranked_ctxs.append({
                "title": all_ctxs[i][1],
                "text": all_ctxs[i][0],
                "score": scores[j]
            })
            j += 1

        verbose_results.append({
            "question": question,
            "ranked_ctxs": ranked_ctxs
        })

    return verbose_results


def load_single_ranking_data(filepath):
    with open(filepath, 'r', encoding='utf8') as fin:
        raw_data = json.load(fin)
        dataset = []
        for fields in raw_data:
            # id_ = fields["id"]
            question = normalize_question(fields["question"])

            def create_passage(ctx: dict):
                return BiEncoderPassage(ctx["text"], ctx["title"])

            positive_passages = [create_passage(ctx) for ctx in fields["positive_ctxs"]]
            hard_negative_passages = [create_passage(ctx) for ctx in fields["hard_negative_ctxs"]]
            rand_negative_passages = [create_passage(ctx) for ctx in fields["rand_negative_ctxs"]]
            assert len(positive_passages) == 1

            all_ctxs = positive_passages + hard_negative_passages + rand_negative_passages

            dataset.append({
                # "id": id_,
                "question": question,
                "all_ctxs": all_ctxs
            })

    logger.info(f"Read {len(dataset)} examples from {filepath}")
    return dataset


def create_biencoder_input(batch, tensorizer, num_ctxs=100):
    batch_question_tensors, batch_ctx_tensors = [], []
    # batch_Q1_pos_idx, batch_Q2_pos_idx = [], []  # index of positive passages in all batch passages

    for example in batch:
        question = example["question"]
        all_ctxs = example["all_ctxs"]
        # if len(all_ctxs) < num_ctxs:
        #     all_ctxs = all_ctxs + [all_ctxs[-1] for _ in range(num_ctxs - len(all_ctxs))]
        assert len(all_ctxs) == num_ctxs

        ctx_tensors = [
            tensorizer.text_to_tensor(ctx.text, title=ctx.title if ctx.title else None) for ctx in all_ctxs
        ]
        question_tensor = tensorizer.text_to_tensor(question)

        # first two passages should be positive passages for Q1 and Q2, respectively
        # batch_Q1_pos_idx.append(len(batch_ctx_tensors))
        # batch_Q2_pos_idx.append(len(batch_ctx_tensors) + 1)

        batch_question_tensors.append(question_tensor)
        batch_ctx_tensors.extend(ctx_tensors)

    batch_question_tensors = torch.stack(batch_question_tensors, dim=0)  # (batch, seq_len)
    batch_ctx_tensors = torch.stack(batch_ctx_tensors, dim=0)  # (batch * 100, seq_len)
    # batch_Q1_pos_idx = torch.LongTensor(batch_Q1_pos_idx)  # (batch,)
    # batch_Q2_pos_idx = torch.LongTensor(batch_Q2_pos_idx)  # (batch,)

    return {
        "question": batch_question_tensors,
        "ctx": batch_ctx_tensors,
        # "Q1_pos_idx": batch_Q1_pos_idx,
        # "Q2_pos_idx": batch_Q2_pos_idx
    }


def validate_single_ranking(cfg, dataset, model, tensorizer):
    logger.info("Starting Ranking Validation ...")
    batch_size = cfg.batch_size
    total_num_batches = math.ceil(len(dataset) / cfg.batch_size)
    rank_scores = []
    rank_results = []

    for batch_i in tqdm(range(total_num_batches), desc="Embedding"):
        batch = dataset[batch_i * batch_size: (batch_i + 1) * batch_size]
        batch = create_biencoder_input(batch, tensorizer, num_ctxs=cfg.num_ctxs)

        device = cfg.device

        questions = batch["question"].to(device)  # tensor, (batch, seq_len)
        all_ctxs = batch["ctx"]  # tensor, (batch * 100, seq_len)
        # first_pos_idx = batch["Q1_pos_idx"]  # tensor, (batch,)
        # second_pos_idx = batch["Q2_pos_idx"]  # tensor, (batch,)
        assert all_ctxs.shape[0] % questions.shape[0] == 0
        ctx_per_question = all_ctxs.shape[0] // questions.shape[0]
        # logger.info(f'Number of candidate passages per question (ctx_per_question): {ctx_per_question}')

        attn_mask = tensorizer.get_attn_mask(questions).long()
        segments = torch.zeros_like(questions)

        # question_emb: (batch, d_model)
        with torch.no_grad():
            _, question_emb, _ = model.question_model(input_ids=questions,
                                                      attention_mask=attn_mask,
                                                      token_type_ids=segments)

        total_ctx_batches = math.ceil(len(all_ctxs) / batch_size)
        all_ctx_embeddings = []
        for ctx_batch_i in range(total_ctx_batches):
            ctx_batch = all_ctxs[ctx_batch_i * batch_size: (ctx_batch_i + 1) * batch_size]  # (batch, seq_len)
            ctx_batch = ctx_batch.to(device)
            ctx_attn_mask = tensorizer.get_attn_mask(ctx_batch).long()
            ctx_segments = torch.zeros_like(ctx_batch)

            with torch.no_grad():
                _, ctx_embedding, _ = model.ctx_model(input_ids=ctx_batch,
                                                      attention_mask=ctx_attn_mask,
                                                      token_type_ids=ctx_segments)

            all_ctx_embeddings.append(ctx_embedding)

        all_ctx_embeddings = torch.cat(all_ctx_embeddings, dim=0)
        assert all_ctx_embeddings.shape[0] == all_ctxs.shape[0]

        all_ctx_embeddings = torch.split(all_ctx_embeddings, ctx_per_question,
                                         dim=0)  # a list of tensors, each (100, d_model)
        question_emb = question_emb.split(1, dim=0)  # a list of tensors, each (1, d_model)

        for idx, (question, ctxs) in enumerate(zip(question_emb, all_ctx_embeddings)):
            question_ctx_sim = dot_product_scores(q_vectors=question, ctx_vectors=ctxs).squeeze().cpu()  # (20,)
            values, indices = torch.sort(question_ctx_sim, dim=0, descending=True)
            rank_scores.append(values.tolist())
            rank_results.append(indices)

    MR, MRR = calculate_MR(rank_results, POS_IDX)
    logger.info(f"Ranking: MR {MR}, MRR {MRR}, {ctx_per_question} passages in total")

    # Output the ranked passages
    if cfg.out_file is not None:
        verbose_results = rank_passages(dataset, rank_results, rank_scores)
        json.dump(verbose_results, open(cfg.out_file, 'w', encoding='utf8'), ensure_ascii=False, indent=4)
        logger.info(f'Saved {len(verbose_results)} ranking results to {cfg.out_file}')


@hydra.main(config_path="conf", config_name="validate_ranking")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    cfg.cuda = not cfg.no_cuda
    logger.info("CFG (after gpu configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, biencoder, _ = init_biencoder_components('hf_bert', cfg, inference_only=True)
    biencoder, _ = setup_for_distributed_mode(biencoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    biencoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(biencoder)
    logger.info("Loading saved model state ...")

    question_encoder_prefix = "question_model."
    context_encoder_prefix = "ctx_model."
    logger.info(f"Question encoder state prefix {question_encoder_prefix}")
    logger.info(f"Context encoder state prefix {context_encoder_prefix}")

    model_to_load.load_state_dict(saved_state.model_dict, strict=True)
    if cfg.cuda:
        biencoder = biencoder.cuda()

    # get questions & answers
    assert cfg.test_file is not None
    dataset = load_single_ranking_data(filepath=cfg.test_file)
    validate_single_ranking(cfg, dataset, biencoder, tensorizer)


if __name__ == "__main__":
    main()