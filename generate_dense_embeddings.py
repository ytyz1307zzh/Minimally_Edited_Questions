"""
Encode all passages in Wikipedia with the trained DPR model.
Supposed to be used in a 'sharded' way to speed up the process.
Code adapted from https://github.com/facebookresearch/DPR.
"""
import logging
import math
import os
import sys
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)
ALL_PASSAGES = 21015324  # The number of wiki passages in DPR index


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logger.info(f"[GPU {cfg.local_rank}] Encoded passages {total}")
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.info("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state)

    logger.info("reading data source: %s", cfg.ctx_src)

    if cfg.total_passages is not None:
        total_num_passages = int(cfg.total_passages)
    else:
        total_num_passages = ALL_PASSAGES
    shard_size = math.ceil(total_num_passages / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_num_passages)
    logger.info(
        f"[GPU {cfg.local_rank}] Passage in this shard: "
        f"{start_idx} to {end_idx} (out of total {total_num_passages})"
    )

    if cfg.distributed_world_size > 1:
        assert cfg.local_rank >= 0
        shard_per_gpu = math.ceil(shard_size / cfg.distributed_world_size)
        start_idx = start_idx + (shard_per_gpu * cfg.local_rank)
        end_idx = min(start_idx + shard_per_gpu, end_idx)
        logger.info(
            f"[GPU {cfg.local_rank}] Passage in this GPU: "
            f"{start_idx} to {end_idx} (out of total {total_num_passages})"
        )

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    shard_passage_dict = {}
    ctx_src.load_data_to(shard_passage_dict, start_idx, end_idx)
    shard_passages = [(k, v) for k, v in shard_passage_dict.items()]
    logger.info(f"[GPU {cfg.local_rank}] Loaded {len(shard_passages)} passages in this shard")

    logger.info(
        f"[GPU {cfg.local_rank}] Producing encodings for passages range: "
        f"{start_idx} to {end_idx} (out of total {total_num_passages})"
    )

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, True)

    if cfg.distributed_world_size > 1:
        file = cfg.out_file + "_" + str(cfg.shard_id) + "_" + str(cfg.local_rank)
    else:
        file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info(f"[GPU {cfg.local_rank}] Writing results to {file}")
    with open(file, mode="wb") as f:
        pickle.dump(data, f, protocol=4)

    logger.info(f"[GPU {cfg.local_rank}] Total passages processed {len(data)}. Written to {file}")


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()
