import os
import sys
import re
import unicodedata
import numpy as np

def normalize_text(tokens):
    """
    Normalize text before WER computation.

    - NFC unicode normalization (e.g. ज़ as 1 codepoint == ज+़ as 2 codepoints;
      without this, decomposed vs. precomposed forms of the same word are
      treated as different words, causing phantom substitutions)
    - lowercase English
    - remove punctuation only
    - preserve Tamil and all other Unicode letters/combining marks
    - collapse multiple spaces
    """
    text = " ".join(tokens)

    # NFC normalization — must happen before punctuation stripping so that
    # base+combining-mark sequences are canonicalized to their composed form
    text = unicodedata.normalize("NFC", text)

    # lowercase English letters
    text = text.lower()

    # remove punctuation only
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )

    # collapse whitespace
    text = " ".join(text.split())

    return text.split()

def build_diff(ref, hyp, path):
    result = []
    # ref = list(map(lambda x: x.lower(), ref))
    # hyp = list(map(lambda x: x.lower(), hyp))
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    r_record = -1
    h_record = -1
    # path = path+[(len(ref), len(hyp))]

    for rpointer, hpointer in path:
        if rpointer!=r_record+1 or hpointer!=h_record+1:
            r_buffer = ' '.join(ref[r_record+1:rpointer])
            r_buffer = r_buffer if len(r_buffer)>0 else "*"
            h_buffer = ' '.join(hyp[h_record+1:hpointer])
            h_buffer = h_buffer if len(h_buffer)>0 else "*"
            result.append(f"({r_buffer}->{h_buffer})")

        result.append(ref[rpointer])
        r_record = rpointer
        h_record = hpointer

    if r_record<len(ref)-1 or h_record<len(hyp)-1:
        r_buffer = ' '.join(ref[r_record+1:])
        r_buffer = r_buffer if len(r_buffer)>0 else "*"
        h_buffer = ' '.join(hyp[h_record+1:])
        h_buffer = h_buffer if len(h_buffer)>0 else "*"
        result.append(f"({r_buffer}->{h_buffer})")
    return ' '.join(result)


# ---------------------------------------------------------------------------
# Confusion-report categorization (report only — never auto-folded into WER)
# ---------------------------------------------------------------------------
# These groups exist purely to bucket *why* a substitution happened, so real
# error counts are never silently hidden. See earlier discussion: retroflex/
# dental/alveolar mixups, nukta/sibilant pairs, cross-script code-mixing,
# digit-vs-word forms, and immediate repeats are each informative on their
# own and should stay in the WER, just labeled.

_RETROFLEX_GROUPS = [
    {"ट", "त"},        # retroflex / dental voiceless stops
    {"ठ", "थ"},        # retroflex / dental aspirated voiceless stops
    {"ड", "द"},        # retroflex / dental voiced stops
    {"ढ", "ध"},        # retroflex / dental aspirated voiced stops
    {"ण", "न"},        # retroflex / dental nasals
    {"ड़", "ढ़", "र"},  # flap consonants / alveolar trill
]

_GRANTHA_PAIRS = [
    {"श", "ष", "स"},   # palatal / retroflex / dental sibilants
    {"क", "क़"},        # nukta pair (velar stop vs. /q/)
    {"ख", "ख़"},        # nukta pair (aspirated velar vs. /x/)
    {"ग", "ग़"},        # nukta pair (velar stop vs. /ɣ/)
    {"ज", "ज़"},        # nukta pair (palatal affricate vs. /z/)
    {"फ", "फ़"},        # nukta pair (aspirated bilabial vs. /f/)
]

_DEVANAGARI_RE = re.compile(r"^[ऀ-ॿ]+$")
_LATIN_RE = re.compile(r"^[a-z0-9]+$")
_DIGIT_RE = re.compile(r"^[0-9]+$")
_DEVANAGARI_DIGIT_RE = re.compile(r"^[०-९]+$")


def _single_char_diff(a, b):
    """If a and b are equal length and differ in exactly one position,
    return (position, char_in_a, char_in_b); else None."""
    if len(a) != len(b):
        return None
    diffs = [(idx, ca, cb) for idx, (ca, cb) in enumerate(zip(a, b)) if ca != cb]
    if len(diffs) != 1:
        return None
    return diffs[0]


def _is_retroflex_confusion(ref_w, hyp_w):
    d = _single_char_diff(ref_w, hyp_w)
    if not d:
        return False
    _, ca, cb = d
    return any(ca in g and cb in g for g in _RETROFLEX_GROUPS)


def _is_nukta_sibilant_confusion(ref_w, hyp_w):
    d = _single_char_diff(ref_w, hyp_w)
    if not d:
        return False
    _, ca, cb = d
    return any(ca in g and cb in g for g in _GRANTHA_PAIRS)


def _is_cross_script(ref_w, hyp_w):
    ref_deva, hyp_deva = bool(_DEVANAGARI_RE.match(ref_w)), bool(_DEVANAGARI_RE.match(hyp_w))
    ref_latin, hyp_latin = bool(_LATIN_RE.match(ref_w)), bool(_LATIN_RE.match(hyp_w))
    return (ref_deva and hyp_latin) or (ref_latin and hyp_deva)


def _is_digit_vs_word(ref_w, hyp_w):
    ref_digit = bool(_DIGIT_RE.match(ref_w)) or bool(_DEVANAGARI_DIGIT_RE.match(ref_w))
    hyp_digit = bool(_DIGIT_RE.match(hyp_w)) or bool(_DEVANAGARI_DIGIT_RE.match(hyp_w))
    return ref_digit != hyp_digit


def categorize_substitution(ref_w, hyp_w):
    """Return a label for one (ref_word, hyp_word) substitution pair.
    Checked in order; first match wins. Falls back to 'other'.
    digit_vs_word is checked before cross_script because a pure digit
    string (e.g. "10") also matches the Latin-script pattern used for
    cross_script — without this ordering, digit/number-word pairs would
    be misfiled as code-mixing instead of numeral-format mismatches."""
    if _is_retroflex_confusion(ref_w, hyp_w):
        return "retroflex_dental_alveolar"
    if _is_nukta_sibilant_confusion(ref_w, hyp_w):
        return "nukta_sibilant"
    if _is_digit_vs_word(ref_w, hyp_w):
        return "digit_vs_word"
    if _is_cross_script(ref_w, hyp_w):
        return "cross_script"
    return "other"


def is_immediate_repeat(word, seq, idx):
    """True if seq[idx] == seq[idx-1], i.e. this token is an immediate
    repeat of the token right before it in that same sequence."""
    return idx > 0 and 0 <= idx < len(seq) and seq[idx] == seq[idx - 1]


def write_confusion_report(report_path, sub_records, insert_records, counters):
    """sub_records: list of (uid, ref_word, hyp_word, category)
    insert_records: list of (uid, hyp_word, is_repeat)"""
    with open(report_path, "w") as f:
        f.write("==================== Substitution categories ====================\n")
        total_subs = sum(counters["sub_by_cat"].values())
        for cat in ["retroflex_dental_alveolar", "nukta_sibilant", "cross_script",
                    "digit_vs_word", "other"]:
            n = counters["sub_by_cat"].get(cat, 0)
            pct = f"{n * 100.0 / total_subs:.2f}%" if total_subs else "n/a"
            f.write(f"  {cat:28s}: {n:6d}  ({pct} of substitutions)\n")
        f.write(f"  {'TOTAL substitutions':28s}: {total_subs:6d}\n")

        f.write("\n==================== Immediate-repeat insertions ====================\n")
        f.write(f"  insertions that repeat previous token : {counters['repeat_insertions']}\n")
        f.write(f"  total insertions                      : {counters['total_insertions']}\n")

        f.write("\n==================== Per-utterance flagged items ====================\n")
        f.write("(uid, category, ref_word->hyp_word)\n")
        for uid, ref_w, hyp_w, cat in sub_records:
            if cat != "other":
                f.write(f"{uid}\t{cat}\t{ref_w}->{hyp_w}\n")
        for uid, hyp_w, is_rep in insert_records:
            if is_rep:
                f.write(f"{uid}\timmediate_repeat_insertion\t*->{hyp_w}\n")


def compute_wer(ref_file,
                hyp_file,
                cer_detail_file,
                confusion_report=False,
                confusion_report_file=None):
    rst = {
        'Wrd': 0,
        'Corr': 0,
        'Ins': 0,
        'Del': 0,
        'Sub': 0,
        'Snt': 0,
        'Err': 0.0,
        'S.Err': 0.0,
        'wrong_words': 0,
        'wrong_sentences': 0
    }

    hyp_dict = {}
    ref_dict = {}
    with open(hyp_file, 'r') as hyp_reader:
        for line in hyp_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            hyp_dict[key] = value
    with open(ref_file, 'r') as ref_reader:
        for line in ref_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            ref_dict[key] = value

    # Only built up when confusion_report is requested; kept out of the hot
    # path otherwise so normal WER scoring is unaffected.
    sub_records = []
    insert_records = []
    counters = {"sub_by_cat": {}, "repeat_insertions": 0, "total_insertions": 0}

    cer_detail_writer = open(cer_detail_file, 'w')
    for hyp_key in hyp_dict:
        if hyp_key in ref_dict:
            out_item = compute_wer_by_line(
                hyp_dict[hyp_key], ref_dict[hyp_key],
                track_details=confusion_report,
            )
            # if out_item['ins'] > 10 or out_item['del'] > 10:
            #     print(hyp_key + print_cer_detail(out_item))
            #     print("ref:" + '\t' + " ".join(list(map(lambda x: x.lower(), ref_dict[hyp_key]))))
            #     print("hyp:" + '\t' + " ".join(list(map(lambda x: x.lower(), hyp_dict[hyp_key]))))
            rst['Wrd'] += out_item['nwords']
            rst['Corr'] += out_item['cor']
            rst['wrong_words'] += out_item['wrong']
            rst['Ins'] += out_item['ins']
            rst['Del'] += out_item['del']
            rst['Sub'] += out_item['sub']
            rst['Snt'] += 1
            if out_item['wrong'] > 0:
                rst['wrong_sentences'] += 1
            cer_detail_writer.write(hyp_key + print_cer_detail(out_item) + '\n')
            # cer_detail_writer.write("ref:" + '\t' + " ".join(list(map(lambda x: x.lower(), ref_dict[hyp_key]))) + '\n')
            cer_detail_writer.write(
                "ref:\t" + " ".join(normalize_text(ref_dict[hyp_key])) + "\n"
            )
            # cer_detail_writer.write("hyp:" + '\t' + " ".join(list(map(lambda x: x.lower(), hyp_dict[hyp_key]))) + '\n')
            cer_detail_writer.write(
                "hyp:\t" + " ".join(normalize_text(hyp_dict[hyp_key])) + "\n"
            )
            cer_detail_writer.write("diff:" + '\t' + build_diff(ref_dict[hyp_key], hyp_dict[hyp_key], out_item['path']) + '\n')

            if confusion_report:
                for ref_w, hyp_w in out_item.get('sub_pairs', []):
                    cat = categorize_substitution(ref_w, hyp_w)
                    counters["sub_by_cat"][cat] = counters["sub_by_cat"].get(cat, 0) + 1
                    sub_records.append((hyp_key, ref_w, hyp_w, cat))
                for hyp_w, is_rep in out_item.get('insert_words', []):
                    counters["total_insertions"] += 1
                    if is_rep:
                        counters["repeat_insertions"] += 1
                    insert_records.append((hyp_key, hyp_w, is_rep))

    if rst['Wrd'] > 0:
        # rst['Err'] = round(rst['wrong_words'] * 100 / rst['Wrd'], 2)
        rst['Err'] = round(
            int(rst['wrong_words']) * 100.0 / int(rst['Wrd']),
            2
        )
    if rst['Snt'] > 0:
        rst['S.Err'] = round(rst['wrong_sentences'] * 100 / rst['Snt'], 2)

    cer_detail_writer.write('\n')
    cer_detail_writer.write("%WER " + str(rst['Err']) + " [ " + str(rst['wrong_words'])+ " / " + str(rst['Wrd']) +
                            ", " + str(rst['Ins']) + " ins, " + str(rst['Del']) + " del, " + str(rst['Sub']) + " sub ]" + '\n')
    cer_detail_writer.write("%SER " + str(rst['S.Err']) + " [ " + str(rst['wrong_sentences']) + " / " + str(rst['Snt']) + " ]" + '\n')
    cer_detail_writer.write("Scored " + str(len(hyp_dict)) + " sentences, " + str(len(hyp_dict) - rst['Snt']) + " not present in hyp." + '\n')
    cer_detail_writer.close()

    if confusion_report:
        report_path = confusion_report_file or (cer_detail_file + ".confusions.txt")
        write_confusion_report(report_path, sub_records, insert_records, counters)
        print(f"Confusion report written to: {report_path}")


def compute_wer_by_line(hyp,
                        ref,
                        track_details=False):
    # hyp = list(map(lambda x: x.lower(), hyp))
    # ref = list(map(lambda x: x.lower(), ref))
    hyp = normalize_text(hyp)
    ref = normalize_text(ref)
    len_hyp = len(hyp)
    len_ref = len(ref)

    # cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int32)
    # cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=int)

    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    sub_pairs = []       # (ref_word, hyp_word) for each substitution, if tracked
    insert_words = []    # (hyp_word, is_immediate_repeat) for each insertion, if tracked
    i = len_hyp
    j = len_ref
    rst = {
        'nwords': len_ref,
        'cor': 0,
        'wrong': 0,
        'ins': 0,
        'del': 0,
        'sub': 0,
        'path': []
    }
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                rst['cor'] += 1

            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            if track_details and i - 1 >= 0:
                insert_words.append((hyp[i - 1], is_immediate_repeat(hyp[i - 1], hyp, i - 1)))
            i -= 1
            rst['ins'] += 1

        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            rst['del'] += 1

        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            if track_details and i - 1 >= 0 and j - 1 >= 0:
                sub_pairs.append((ref[j - 1], hyp[i - 1]))
            i -= 1
            j -= 1
            rst['sub'] += 1

        if i < 0 and j >= 0:
            rst['del'] += 1
        elif j < 0 and i >= 0:
            rst['ins'] += 1

    match_idx.reverse()
    sub_pairs.reverse()
    insert_words.reverse()
    # wrong_cnt = cost_matrix[len_hyp][len_ref]
    # rst['wrong'] = wrong_cnt
    wrong_cnt = int(cost_matrix[len_hyp][len_ref])
    rst['wrong'] = wrong_cnt
    rst['path'] = match_idx
    rst['cor'] = int(rst['cor'])
    rst['ins'] = int(rst['ins'])
    rst['del'] = int(rst['del'])
    rst['sub'] = int(rst['sub'])
    if track_details:
        rst['sub_pairs'] = sub_pairs
        rst['insert_words'] = insert_words
    return rst

def print_cer_detail(rst):
    return ("(" + "nwords=" + str(rst['nwords']) + ",cor=" + str(rst['cor'])
            + ",ins=" + str(rst['ins']) + ",del=" + str(rst['del']) + ",sub="
            + str(rst['sub']) + ") corr:" + '{:.2%}'.format(rst['cor']/rst['nwords'])
            + ",cer:" + '{:.2%}'.format(rst['wrong']/rst['nwords']))

if __name__ == '__main__':
    argv = sys.argv[1:]

    confusion_report = "--confusion-report" in argv
    if confusion_report:
        argv.remove("--confusion-report")

    confusion_report_file = None
    if "--confusion-out" in argv:
        idx = argv.index("--confusion-out")
        confusion_report_file = argv[idx + 1]
        del argv[idx:idx + 2]

    if len(argv) != 3:
        print("usage : python indic_normalise_werNFC.py test.ref test.hyp test.wer "
              "[--confusion-report] [--confusion-out path]")
        print()
        print("  --confusion-report   also write a substitution/insertion breakdown")
        print("                       (retroflex/dental/alveolar, nukta/sibilant,")
        print("                       cross-script, digit-vs-word, immediate-repeat)")
        print("                       instead of folding these into plain WER.")
        print("                       Off by default.")
        print("  --confusion-out PATH override the confusion-report output path")
        print("                       (default: <test.wer>.confusions.txt)")
        sys.exit(0)

    ref_file, hyp_file, cer_detail_file = argv
    compute_wer(ref_file, hyp_file, cer_detail_file,
                confusion_report=confusion_report,
                confusion_report_file=confusion_report_file)
