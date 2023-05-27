data=$1  # dataset name
model=$2  # model checkpoint

if [ "$data" == "nq-train-ambigqa" ]; then
    data_path=dataset/ranking/ambigqa-nq-train-ranking.json
elif [ "$data" == "ambigqa" ]; then
    data_path=dataset/ranking/ambigqa-ranking.json
elif [ "$data" == "nq-test" ]; then
    data_path=dataset/ranking/nq-test-ranking.json
elif [ "$data" == "nq-train-gpt" ]; then
    data_path=dataset/ranking/gpt-nq-train-ranking.json
elif [ "$data" == "gpt" ]; then
    data_path=dataset/ranking/gpt-ranking.json
else
    echo "Invalid argument ${data}"
    exit 1
fi

option="
    test_file=${data_path}
    model_file=${model}
    batch_size=128
    num_ctxs=50
    out_file=ranking_results_${data}.json
"
# encoder.pretrained_model_cfg=bert-large-uncased

cmd="python validate_ranking.py ${option}"

echo $cmd
eval $cmd