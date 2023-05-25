"""
Analyse DPR retrieval results
Input: DPR retrieval file, with all questions (training set questions + contrastive test set questions)
Evaluation: Hit@k on all questions, Hit@k on individual sets, retrieval passage overlap
"""
import json
import argparse


def mean(array):
    return sum(array) / len(array)


def calculate_hit_accuracy(retrieval_data):
    """
    Calculate Hit@k of the retrieved data
    """
    hit1, hit5, hit20, hit100 = [], [], [], []

    for instance in retrieval_data:
        passages = instance["ctxs"]
        has_answer = [p["has_answer"] for p in passages]
        hit1.append(int(has_answer[0]))
        hit5.append(int(any(has_answer[:5])))
        hit20.append(int(any(has_answer[:20])))
        hit100.append(int(any(has_answer)))

    print(f"Hit@1: {mean(hit1)}")
    print(f"Hit@5: {mean(hit5)}")
    print(f"Hit@20: {mean(hit20)}")
    print(f"Hit@100: {mean(hit100)}")


def calculate_positive_mrr(retrieval_data):
    """
    Calculate the mean reciprocal rank for the first retrieved positive passage
    """
    rank_list = [None for _ in range(len(retrieval_data))]

    for i in range(len(retrieval_data)):
        instance = retrieval_data[i]
        passage_list = instance["ctxs"]
        for j in range(len(passage_list)):
            passage = passage_list[j]
            # Found the first positive passage
            if passage["has_answer"]:
                rank_list[i] = j + 1
                break

    reciprocal_rank_list = [1 / x if x is not None else 0 for x in rank_list]
    print(f'MRR: {mean(reciprocal_rank_list)}')


def analyse_passage_overlap(retrieval_data):
    """
    Analyse how much do the retrieved passages of two similar questions overlap
    """
    num_instances = int(len(retrieval_data) / 2)  # The source data is in pairs

    total_top5_overlap = []
    total_top20_overlap = []
    total_top100_overlap = []

    for i in range(num_instances):
        first_retrieval = retrieval_data[2 * i]
        second_retrieval = retrieval_data[2 * i + 1]

        first_passages = first_retrieval["ctxs"]
        second_passages = second_retrieval["ctxs"]

        first_passage_ids = [x["id"] for x in first_passages]
        second_passage_ids = [x["id"] for x in second_passages]

        # The retrieval results should be clear of duplicates
        assert len(set(first_passage_ids)) == len(first_passage_ids)
        assert len(set(second_passage_ids)) == len(second_passage_ids)

        assert len(first_passage_ids) == len(second_passage_ids)
        top5_overlap = set(first_passage_ids[:5]).intersection(set(second_passage_ids[:5]))
        top20_overlap = set(first_passage_ids[:20]).intersection(set(second_passage_ids[:20]))
        top100_overlap = set(first_passage_ids).intersection(set(second_passage_ids))

        total_top5_overlap.append(len(top5_overlap))
        total_top20_overlap.append(len(top20_overlap))
        total_top100_overlap.append(len(top100_overlap))

    print(f'Average top5 overlap: {mean(total_top5_overlap)}')
    print(f'Average top20 overlap: {mean(total_top20_overlap)}')
    print(f'Average top100 overlap: {mean(total_top100_overlap)}')


def get_overlap_passage(first_retrieval, second_retrieval, passage_with_both_answers_ids):
    first_question = first_retrieval["question"]
    first_answer = first_retrieval["answers"]
    second_question = second_retrieval["question"]
    second_answer = second_retrieval["answers"]

    passage_details = []
    for passage_info in first_retrieval["ctxs"]:
        if passage_info["id"] in passage_with_both_answers_ids:
            passage_details.append({
                "id": passage_info["id"],
                "title": passage_info["title"],
                "text": passage_info["text"]
            })

    return {
        "Q1": first_question,
        "A1": first_answer,
        "Q2": second_question,
        "A2": second_answer,
        "passages": passage_details
    }


parser = argparse.ArgumentParser()
parser.add_argument('-retrieval', type=str, default='data/gpt3_filter/nq-train/dpr-output/contriever-0.95-0.99-retrieval.json',
                    help='Path to the file containing retrieved passages')
parser.add_argument('-passage_overlap_outpath', type=str, default=None,
                    help='If not None, the retrieved passages with answers of both questions are saved in this path')
args = parser.parse_args()

retrieval_data = json.load(open(args.retrieval, 'r', encoding='utf8'))
print(f'Load {len(retrieval_data)} data instances ({len(retrieval_data) // 2} question pairs)')

calculate_hit_accuracy(retrieval_data)
calculate_positive_mrr(retrieval_data)
first_question_retrieval = [retrieval_data[2 * i] for i in range(len(retrieval_data) // 2)]
second_question_retrieval = [retrieval_data[2 * i + 1] for i in range(len(retrieval_data) // 2)]
print('\nFirst set of questions:')
calculate_hit_accuracy(first_question_retrieval)
calculate_positive_mrr(first_question_retrieval)
print('\nSecond set of questions:')
calculate_hit_accuracy(second_question_retrieval)
calculate_positive_mrr(second_question_retrieval)
print()

analyse_passage_overlap(retrieval_data)
print()
