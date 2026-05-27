import json
import time
import random
import argparse

from utils import Container
from typing import List, Optional, Tuple
from datetime import datetime
from guardrail import guard_rail
from prompts import WEB_AGENT_SPECIFICATION, WEB_AGENT_USAGE_PRINCEPLE


QUERY_KEYS = ("attack_query", "carrier_query", "prohibited_query", "host_query", "query")


def get_query(record):
    for key in QUERY_KEYS:
        if key in record:
            return record[key]
    raise KeyError(f"Missing query field. Expected one of: {', '.join(QUERY_KEYS)}")


def kendall_inversions(perm: List[int]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            inv += (perm[i] > perm[j])
    return inv


def adjacent_kept(perm: List[int]) -> int:
    # Count how many adjacent index relationships are preserved.
    pos = {v: idx for idx, v in enumerate(perm)}
    kept = 0
    for i in range(len(perm) - 1):
        kept += (pos[i + 1] == pos[i] + 1)
    return kept


def monotone_runs(perm: List[int]) -> int:
    # Count monotonic runs. More runs means a less globally ordered sequence.
    if len(perm) <= 2:
        return 1
    runs = 1
    prev = 0
    for i in range(1, len(perm)):
        cur = 1 if perm[i] > perm[i - 1] else (-1 if perm[i] < perm[i - 1] else 0)
        if cur != 0 and prev != 0 and cur != prev:
            runs += 1
        if cur != 0:
            prev = cur
    return runs


def best_high_disruption_permutation(k: int, seed: Optional[int] = None, trials: int = 20000) -> List[int]:
    """
    Return a high-disruption permutation over indices 0..k-1.
    Larger trials increase the chance of finding a more disordered permutation.
    """
    rng = random.Random(seed)
    max_inv = k * (k - 1) // 2

    best = None
    best_score = None

    for _ in range(trials):
        perm = list(range(k))
        rng.shuffle(perm)

        inv = kendall_inversions(perm)
        adj = adjacent_kept(perm)
        runs = monotone_runs(perm)

        # Normalize.
        inv_ratio = inv / max_inv if max_inv else 0.0

        # Score high inversions, low adjacent preservation, and many runs.
        # Tune weights as needed.
        score = (
            3.0 * inv_ratio
            - 2.0 * (adj / max(1, k - 1))
            + 1.0 * (runs / max(1, k - 1))
        )

        if best_score is None or score > best_score:
            best_score = score
            best = perm

    return best


def apply_word_permutation(words: List[str], perm: List[int]) -> List[str]:
    return [words[i] for i in perm]


def insert_keep_word_order_allow_same_position(
    sentence: str,
    words: List[str],
    seed: Optional[int] = None) -> str:
    tokens = sentence.split()
    n = len(tokens)
    positions = n + 1  # 0..n, including beginning and end.
    k = len(words)

    rng = random.Random(seed)

    # Generate random positions with duplicates allowed, then sort them.
    pos_seq = [rng.randrange(positions) for _ in range(k)]
    pos_seq.sort()

    # buckets[pos] stores words inserted at that position in original order.
    buckets: List[List[str]] = [[] for _ in range(positions)]
    for w, pos in zip(words, pos_seq):
        buckets[pos].append(w)

    out: List[str] = []

    # Position 0: sentence start.
    for w in buckets[0]:
        out.append(f"<{w}>")

    # Token followed by the bucket after it; position i+1 is after token[i].
    for i, tok in enumerate(tokens):
        out.append(tok)
        for w in buckets[i + 1]:
            out.append(f"<{w}>")

    return " ".join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--guardrail_model', type=str, default='gpt-4o')
    parser.add_argument('--file', type=str, default='../datasets/SeeAct/sample_labeled_benign.json')
    args = parser.parse_args()

    safety_ceritera = {
        "Query": "query",
        "Rule-based Checking": "You need to check the related rules according to the agent usage principles."
    }

    tools = {
        "": ""
    }

    container = Container()
    crt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    start = time.perf_counter()

    # results = []
    for i in range(10):
        case = data[i]
        user_info = case["user_info"]
        user_info["user_task"] = get_query(case)
        # user_info["website"] = case["website"]
        print(json.dumps(user_info, indent=4))
        # break

        guardrail_output, check_process, code, length = guard_rail(tools=tools,
                                                                user_information=user_info,
                                                                agent_action="",
                                                                agent_specification=WEB_AGENT_SPECIFICATION,
                                                                agent_usage_principles=WEB_AGENT_USAGE_PRINCEPLE,
                                                                memory=safety_ceritera,
                                                                retrieve_file="",
                                                                environment=container,
                                                                agent_model=args.guardrail_model,
                                                                update=False)
        print(guardrail_output)
        print(i)
        
        # results.append(
        new_res = {
                "user_info": json.dumps(user_info),
                "annotation_id": case["annotation_id"],
                "label": case["label"],
                "DAS": str(guardrail_output),
                "code": str(code),
                "check": json.dumps(check_process)
            }
        # )
        with open("result/seeact/{0}.json".format(crt), "a") as f:
            json.dump(new_res, f, indent = 4)
            f.write("\n---------------------------------------\n")

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")

    container.__del__()
