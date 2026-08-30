#!/usr/bin/env python3
"""Manual WER (word-level Levenshtein) on SLAM-LLM decode files. Dependency-free."""
import argparse
import re
from pathlib import Path

DEFAULT_PRED = "/speech/tomson/exps/speech-recog/exps/data2vec-aqc-hindi-sarvam1-4000hours-20260528_171346/asr_epoch_1_step_63000/decode_hindi_indictts_test_beam1_pred"
DEFAULT_GT = "/speech/tomson/exps/speech-recog/exps/data2vec-aqc-hindi-sarvam1-4000hours-20260528_171346/asr_epoch_1_step_63000/decode_hindi_indictts_test_beam1_gt"


def normalize(text):
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_line(line):
    line = line.rstrip("\r\n")
    if "\t" in line:
        return line.split("\t", 1)
    parts = line.split(None, 1)
    return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")


def load(path):
    d, order = {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            uid, txt = parse_line(line)
            if uid not in d:
                order.append(uid)
            d[uid] = txt
    return d, order


def align_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    S = I = D = H = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            H += 1; i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, I, D, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--gt", default=DEFAULT_GT)
    args = ap.parse_args()

    pred, pred_order = load(Path(args.pred))
    gt, _ = load(Path(args.gt))
    common = [u for u in pred_order if u in gt]
    if not common:
        print("No overlapping utterance ids between pred and gt yet.")
        return

    tot_S = tot_I = tot_D = tot_H = 0
    ref_words = hyp_words = sentences = empty_refs = 0
    for u in common:
        ref = normalize(gt[u]).split()
        hyp = normalize(pred[u]).split()
        if not ref:
            empty_refs += 1
            continue
        S, I, D, H = align_counts(ref, hyp)
        tot_S += S; tot_I += I; tot_D += D; tot_H += H
        ref_words += len(ref); hyp_words += len(hyp); sentences += 1

    N = ref_words
    errors = tot_S + tot_I + tot_D
    wer = errors / N if N else float("nan")

    print("==================== WER (manual edit-distance) ====================")
    print(f"  pred file          : {args.pred}")
    print(f"  gt   file          : {args.gt}")
    print(f"  pred lines         : {len(pred)}")
    print(f"  gt   lines         : {len(gt)}")
    if empty_refs:
        print(f"  empty refs skipped : {empty_refs}")
    print("--------------------------------------------------------------------")
    print(f"  Sentences          : {sentences}")
    print(f"  Reference words  N : {N}")
    print(f"  Hypothesis words   : {hyp_words}")
    print(f"  Substitutions    S : {tot_S}")
    print(f"  Insertions       I : {tot_I}")
    print(f"  Deletions        D : {tot_D}")
    print(f"  Hits             H : {tot_H}")
    print(f"  Errors   S+I+D     : {errors}")
    print("--------------------------------------------------------------------")
    print(f"  WER = (S+I+D)/N    : {wer * 100:.2f}%")
    print("====================================================================")


if __name__ == "__main__":
    main()
