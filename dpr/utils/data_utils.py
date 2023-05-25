import json
import logging
import pdb
import pickle
import random

import itertools
import math
import time
import torch
from torch import Tensor as T
from typing import List, Iterator, Callable, Tuple

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_jsonl_files(paths: List[str], local_rank: int):

    assert len(paths) == 1
    path = paths[0]
    logging.info(f'Loading {path}')

    results = []
    logger.info(f"[GPU {local_rank}] Reading file {path}")
    for ln, line in enumerate(open(path, 'r', encoding='utf8')):
        results.append(json.loads(line))

    logger.info(f"[GPU {local_rank}] Aggregated data size: {len(results)}")
    return results


def read_data_from_json_files(paths: List[str], local_rank: int) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            if local_rank > 0:
                logger.info(f"[GPU {local_rank}] Going into sleep for {local_rank * 500} seconds,"
                            f" let other threads go first...")
                time.sleep(local_rank * 500)
                logger.info(f"[GPU {local_rank}] Waking up...")
            logger.info(f"[GPU {local_rank}] Reading file {path}")
            results = json.load(f)
            logger.info(f"[GPU {local_rank}] Aggregated data size: {len(results)}")
    return results


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    logger.info(f'Read {len(result)} data from {path}')
    return result


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
        batched_example_idx: str = None,
    ):

        self.data = data

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        def calculate_total_size_from_batch(batched_example_idx):
            batches_first_epoch = batched_example_idx[0]
            total_num_examples = 0
            for batch in batches_first_epoch:
                total_num_examples += len(batch)
            return total_num_examples

        self.batched_example_idx = None
        if batched_example_idx is not None:
            self.batched_example_idx = read_jsonl_as_list(batched_example_idx)
            self.total_size = calculate_total_size_from_batch(self.batched_example_idx)
            logger.warning(f"Using pre-computed batches")
        else:
            logger.warning(f"Using random batching paradigm")
            self.total_size = len(data)

        samples_per_shard = math.ceil(self.total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, self.total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
        )

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return self.total_size

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        return self.max_iterations

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        # If there exists a pre-defined list of batches, directly use them
        if self.batched_example_idx is not None:
            batched_idxs_this_epoch = self.batched_example_idx[epoch]  # (num_batches, batch * shards)
            # If the last batch is a small batch, then DPR will not iterate that last one
            if len(batched_idxs_this_epoch[-1]) == len(batched_idxs_this_epoch[-2]):
                num_batches = len(batched_idxs_this_epoch)
            else:
                num_batches = len(batched_idxs_this_epoch) - 1
            logger.info(f'Total number of batches in batched_example_idx: {num_batches}')
            shard_indices = []
            for batch_i in range(num_batches):
                batch_idxs = batched_idxs_this_epoch[batch_i]
                assert len(batch_idxs) % self.shards_num == 0
                single_shard_size = len(batch_idxs) // self.shards_num
                shard_indices_this_batch = batch_idxs[
                                           self.shard_id * single_shard_size: (self.shard_id + 1) * single_shard_size
                                           ]
                shard_indices.extend(shard_indices_this_batch)
        else:
            indices = list(range(len(self.data)))
            if self.shuffle:
                # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
                epoch_rnd = random.Random(self.shuffle_seed + epoch)
                epoch_rnd.shuffle(indices)
            shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    # TODO: merge with iterate_ds_sampled_data
    # def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
    #     # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
    #     max_iterations = self.max_iterations - self.iteration
    #     shard_indices = self.get_shard_indices(epoch)
    #
    #     for i in range(self.iteration * self.batch_size, len(shard_indices), self.batch_size):
    #         items_idxs = shard_indices[i : i + self.batch_size]
    #         if self.strict_batch_size and len(items_idxs) < self.batch_size:
    #             logger.debug("Extending batch to max size")
    #             items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
    #         self.iteration += 1
    #         items = [self.data[idx] for idx in items_idxs]
    #         yield items
    #
    #     # some shards may done iterating while the others are at the last batch. Just return the first batch
    #     while self.iteration < max_iterations:
    #         logger.debug("Fulfilling non complete shard=".format(self.shard_id))
    #         self.iteration += 1
    #         items_idxs = shard_indices[0 : self.batch_size]
    #         items = [self.data[idx] for idx in items_idxs]
    #         yield items
    #
    #     logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
    #     # reset the iteration status
    #     self.iteration = 0

    def iterate_ds_sampled_data(self, num_iterations: int, epoch: int = 0) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1

            # In training, each index is a dict of data id, positive question id, negative question id
            if isinstance(items_idxs[0], dict):
                items = [self.data.get_item(
                    data_idx=idx["id"],
                    pos_question_idx=idx["pos_id"],
                    hard_neg_question_idx=idx["neg_id"]
                ) for idx in items_idxs]
            # In validation, each index is only data id
            else:
                items = [self.data.get_item(data_idx=idx) for idx in items_idxs]

            if self.iteration == 1:
                logger.info(f"[GPU {self.shard_id}] first batch indices: {items_idxs}")
            yield items

        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # TODO: reset the iteration status?
        self.iteration = 0

    def get_dataset(self) -> torch.utils.data.Dataset:
        return self.data


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        sampling_rates: List = [],
        rank: int = 0,
    ):
        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info("rank=%d; Multi set data sizes %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            self.max_its_pr_ds = [int(ds.max_iterations_num() * sampling_rates[i]) for i, ds in enumerate(datasets)]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        logger.info("rank=%d; Iteration start", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set: %s",
            self.rank,
            [it.get_iteration() for it in self.iterables],
        )

        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            iterators.append(self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch))

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        logger.info("rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices))
        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning("rank=%d; Next item in the source %s is None", self.rank, source_idx)

        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        logger.info(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def get_dataset(self, ds_id: int) -> torch.utils.data.Dataset:
        return self.iterables[ds_id].get_dataset()

    def get_datasets(self) -> List[torch.utils.data.Dataset]:
        return [it.get_dataset() for it in self.iterables]


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError
