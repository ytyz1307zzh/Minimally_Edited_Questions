LR=1e-5  # learning rate

DATA_DIR=dataset  # directory of the data
OUTPUT_DIR=checkpoints  # directory of saving checkpoints & results
BATCH_IDX=dataset/train/contrastive-augment-33k-train-batches64_idx.jsonl  # pre-computed data indices of each batch

BATCH=16
DEV_BATCH=8
EPOCHS=40
N_GPUS=4
QUESTION_LOSS=$2  # options: contrastive, hinge, dot
HINGE_MARGIN=$3  # the "alpha" in the triplet loss as described in the paper
CONTRAST_LOSS=$4  # the weight of the query-side contrastive loss
CONTRAST_START_EPOCH=$5  # the epoch to start the contrastive loss, 0 means starting from beginning
TOTAL_BATCH=$((BATCH * N_GPUS))

# The directory of saving this run
CKPT_NAME="MEQ_${QUESTION_LOSS}_batch${TOTAL_BATCH}_lr${LR}_epoch${EPOCHS}_loss${CONTRAST_LOSS}_margin${HINGE_MARGIN}_start${CONTRAST_START_EPOCH}"

# DPR model training

option1="
   train_datasets=[nq_contrast_33k_train]
   dev_datasets=[nq_dev_2k,ambigqa_dev,surge_dev]
   batched_example_idx=${BATCH_IDX}
   train=biencoder_nq
   train.num_train_epochs=${EPOCHS}
   train.learning_rate=${LR}
   train.log_batch_step=1000000
   train.question_loss=${QUESTION_LOSS}
   train.contrast_loss_weight=${CONTRAST_LOSS}
   train.hinge_margin=${HINGE_MARGIN}
   train.contrast_start_epoch=${CONTRAST_START_EPOCH}
   train.eval_per_epoch=1
   train.warmup_steps=2000
   train.batch_size=${BATCH}
   train.dev_batch_size=${DEV_BATCH}
   train.val_av_rank_bsz=32
   train.val_av_rank_hard_neg=30
   train.val_av_rank_other_neg=20
   val_av_rank_start_epoch=0
   output_dir=${OUTPUT_DIR}/${CKPT_NAME}
   fp16=False
   seed=46556
"

cmd1="python -m torch.distributed.launch --nproc_per_node=${N_GPUS} train_dense_encoder.py ${option1}"

echo $cmd1
eval $cmd1

# Ranking evaluation

cmd2="CUDA_VISIBLE_DEVICES=0 python validate_single_ranking.py
    test_file=${DATA_DIR}/ranking/nq-test-ranking.json
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_0.model
    batch_size=128
    num_ctxs=50"

echo $cmd2
eval $cmd2


cmd2="CUDA_VISIBLE_DEVICES=0 python validate_single_ranking.py
    test_file=${DATA_DIR}/ranking/ambigqa-ranking.json
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_1.model
    batch_size=128
    num_ctxs=50"

echo $cmd2
eval $cmd2

cmd2="CUDA_VISIBLE_DEVICES=0 python validate_single_ranking.py
    test_file=${DATA_DIR}/ranking/surge-ranking.json
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_2.model
    batch_size=128
    num_ctxs=50"

echo $cmd2
eval $cmd2

# Generate passage embeddings for Wiki passages

option3="
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_2.model
    ctx_src=wikipedia
    batch_size=2048
    shard_id=0
    num_shards=1
    out_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_embedding_2.wikipedia
    fp16=False
"

cmd3="python -m torch.distributed.launch --nproc_per_node=${N_GPUS} generate_dense_embeddings.py ${option3}"

echo $cmd3
eval $cmd3

# Retrieve Wiki passages for each question

option4="
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_2.model
    qa_dataset=surge_test
    ctx_datatsets=[wikipedia]
    encoded_ctx_files=[\"${OUTPUT_DIR}/${CKPT_NAME}/dpr_embedding_2.wikipedia_*\"]
    out_file=${OUTPUT_DIR}/${CKPT_NAME}/surge_test_wikipedia
    batch_size=32
    validation_workers=4
"

cmd4="python dense_retriever.py ${option4}"

echo $cmd4
eval $cmd4

# Evaluate the retrieval results

cmd5="python evaluate/passage_hit_and_overlap.py -retrieval ${OUTPUT_DIR}/${CKPT_NAME}/surge_test_wikipedia_output.json"

echo $cmd5
eval $cmd5

# Repeat steps 3-5 for the AmbigQA contrast set

option3="
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_1.model
    ctx_src=wikipedia
    batch_size=2048
    shard_id=0
    num_shards=1
    out_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_embedding_1.wikipedia
    fp16=False
"

cmd3="python -m torch.distributed.launch --nproc_per_node=${N_GPUS} generate_dense_embeddings.py ${option3}"

echo $cmd3
eval $cmd3

option4="
    model_file=${OUTPUT_DIR}/${CKPT_NAME}/dpr_biencoder_best_1.model
    qa_dataset=ambigqa_test
    ctx_datatsets=[wikipedia]
    encoded_ctx_files=[\"${OUTPUT_DIR}/${CKPT_NAME}/dpr_embedding_1.wikipedia_*\"]
    out_file=${OUTPUT_DIR}/${CKPT_NAME}/ambigqa_test_wikipedia
    batch_size=32
    validation_workers=4
"

cmd4="python dense_retriever.py ${option4}"

echo $cmd4
eval $cmd4

cmd5="python evaluate/passage_hit_and_overlap.py -retrieval ${OUTPUT_DIR}/${CKPT_NAME}/ambigqa_test_wikipedia_output.json"

echo $cmd5
eval $cmd5
