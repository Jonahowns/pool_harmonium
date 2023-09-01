import os
import pickle
import math
import sys
import time

from collections import Counter
from multiprocessing import Pool
from itertools import repeat, chain

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pool.utils.io import fasta_read_basic, gunter_read, csv_read, write_fasta


def enrichment_average(df, label_names, min_diff=1, max_diff=None, diff_weights=None, label_weights=None):
    label_number = len(label_names)

    if max_diff is None:
        max_diff = label_number-1

    if diff_weights is None:
        diff_weights = [1. for _ in range(min_diff, max_diff+1)]

    if label_weights is None:
        label_weights = [1. for _ in range(len(label_names))]

    # first let's remove all the nan values in the dataframe, set nan values as the minimum normalized count for each label
    for r in label_names:
        df[r] = df[r].fillna(df[r].min())

    # Get fold value for label differences
    fold_keys = {diff: [] for diff in range(min_diff, max_diff+1)}
    for i in range(label_number):
        for j in range(label_number):
            if i >= j or j - i < min_diff or j - i > max_diff:
                continue
            fold_column_name = f"fold_{label_names[j]}v{label_names[i]}"
            fold_keys[j-i].append(fold_column_name)
            # fold_diffs.append(j-i)
            df[fold_column_name] = df[label_names[j]]/df[label_names[i]] * (label_weights[j] + label_weights[i])

    diff_keys = []
    for i in range(min_diff, max_diff+1):
        diff_avg_key = f"fold_diff{i}_avg"
        df[diff_avg_key] = df[fold_keys[i]].sum(axis=1).div(len(fold_keys[i])).mul(diff_weights[i-1])
        diff_keys.append(diff_avg_key)

    df["Final_Fold_Avg"] = df[diff_keys].sum(axis=1).div(len(diff_keys))

    return df


def data_properties(seqs, label=None, outfile=sys.stdout, calculate_copy_number=True):
    """ Get Dataframe with Sequence Lengths and Copy Number of each sequence.
     Also creates log """
    if outfile != sys.stdout:
        outfile = open(outfile, 'w+')

    useqs = list(set(seqs))

    if calculate_copy_number:
        cpy_num = Counter(seqs)
        copy_number = [cpy_num[x] for x in useqs]

    print(f'Removed {len(seqs)-len(useqs)} Repeat Sequences', file=outfile)
    ltotal = [len(s) for s in useqs]

    labellist = [label for x in useqs]

    data = {"sequence": useqs, "length": ltotal, "label": labellist, "copy_num": copy_number}

    if label is not None:
        data.update({"label": [label for _ in useqs]})

    if calculate_copy_number:
        data.update({"copy_num": copy_number})

    df = pd.DataFrame(data)

    lp = set(ltotal)
    lps = sorted(lp)
    counts = []
    for x in lps:
        c = 0
        for aas in useqs:
            if len(aas) == x:
                c += 1
        counts.append(c)
        print('Length:', x, 'Number of Sequences', c, file=outfile)

    return df


# Keep sequences of length lmin <= x <= lmax
def prep_data(seqs, lmin=0, lmax=10, cpy_num=None):
    fseqs = []
    affs = []
    for sid, s in enumerate(seqs):
        if lmin <= len(s) <= lmax:
            fseqs.append(s)
            if cpy_num is not None:
                affs.append(cpy_num[sid])
        else:
            continue
    if cpy_num is not None:
        return fseqs, affs
    else:
        return fseqs


def wildcard_replacer(seqs, wildcards=['*']):
    """

    Parameters
    ----------
    seqs: list of str,
        list of sequences
    wildcards: list of strs, optional, default=['*']
        list of wildcard symbols to replace with '-'
    Returns
    -------
    seqs: list of str,
        sequences with wildcards replaced as gaps '-'

    """
    nseqs = []
    for seq in seqs:
        for wc in wildcards:
            seq = seq.replace(wc, '-')
        nseqs.append(seq)
    return nseqs


def gap_adder(seqs, target_length, gap_indx=-1):
    """ Adds gaps to sequences at specified index to match a provided length

      Parameters
    ----------
    seqs: list of str,
        sequences
    target_length: int,
        length all sequences will be standardized to. Does not delete bases of sequences longer than target_length
    gap_indx: int, optional, default=-1
        where to add gaps in the sequence to reach the target length

    Returns
    -------
    nseqs: list of str,
        the input sequences standardized to with gaps added to the new length
    """

    nseqs = []
    for seq in seqs:
        if gap_indx == -1:
            nseqs.append(seq.replace("*", "-") + "".join(["-" for i in range(target_length - len(seq))]))
        else:
            dashes = "".join(["-" for i in range(target_length - len(seq))])
            seq.insert(dashes, gap_indx)
            nseqs.append(seq)
    return nseqs


# Clusters are defined by the length of the sequences
# Extractor takes all data and writes fasta file for sequences of the specified lengths
def extractor(seqs, copy_num, length_indices, outdir='./', uniform_length=True, position_indx=[-1]):
    """
    Write only sequences of specified length to a fasta file with their corresponding copy numbers

    Parameters
    ----------
    seqs: list of str
        the sequences
    copy_num: list of ints or floats
        number of times each sequence was observed in the dataset
    length_indices: list of tuples,
        defines lengths of each cluster ex. [(0, 5), (6, 10)] separates the sequnces
    outdir
    uniform_length
    position_indx

    Returns
    -------
    None

    Writes out each cluster to a fasta file in the directory defined by outdir

    """
    cnum = len(length_indices)
    for i in range(cnum):
        ctmp, caffs = prep_data(seqs, length_indices[i][0], length_indices[i][1], cpy_num=copy_num)
        ctmp = wildcard_replacer(ctmp, wildcards=['*'])
        if uniform_length:
            ctmp = gap_adder(ctmp, length_indices[i][1], position_indx=position_indx[i])
        write_fasta(ctmp, caffs, outdir + f'_c{i}.fasta')


# Should refactor to be process_raw_sequence_files
def process_raw_fasta_files(*files, in_dir="./", out_dir="./", violin_out=None, input_format="fasta"):
    """ Read in Sequence Files and extract relevant information to a pandas dataframe

    Parameters
    ----------
    *files: str
        filenames of all sequence files (include extension)
    in_dir: str, optional, default="./"
        directory where fasta files are stored (relative to where you call this function from)
    out_dir: str, optional, default="./"
        directory where full length reports are saved (relative to where you call this function from)
    violin_out: str, optional, default=None,
        file to save violin plot of data lengths to (do not include file extension)
    input_format: str, optional, default="fasta"
        format of the provided sequence files, supported options {"fasta", "gunter", "caris"}

    Returns
    -------
    dataframe: pandas.DataFrame
        contains all sequence, copy_number, length, and label information from input files
    """
    dfs = []
    all_chars = []  # Find what characters are in dataset
    for file in files:
        rnd = os.path.basename(file).split(".")[0]  # Round is the name of the fasta file
        file = in_dir + file
        if input_format == "fasta":
            seqs, rnd_chars = fasta_read_basic(file)
            all_chars += rnd_chars
            df = data_properties(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=True)
            dfs.append(df)
        elif input_format == "gunter":
            seqs, copy_num, rnd_chars = gunter_read(file)
            all_chars += rnd_chars
            df = data_properties(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=False)
            df["copy_num"] = copy_num
            dfs.append(df)
        elif input_format == "caris":
            seqs, copy_num, rnd_chars = csv_read(file, sequence_label="sequence", copy_num_label="reads")
            all_chars += rnd_chars
            df = data_properties(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=False)
            df["copy_num"] = copy_num
            dfs.append(df)
        else:
            print(f"Input file format {input_format} is not supported.")
            print(exit(-1))

    all_chars = list(set(all_chars))
    master_df = pd.concat(dfs)
    if violin_out is not None:
        sns.violinplot(data=master_df, x="label", y="length")
        violin_out = out_dir + violin_out
        plt.savefig(violin_out+".png", dpi=300)
    print("Observed Characters:", all_chars)
    return master_df


def copynumber_topology(dataframe, labels, threads_per_task=1):
    dfs = []
    for r in labels:
        dfs.append(dataframe[dataframe["label"] == r])
        
    merged_df_pairs = []
    for i in range(0, len(labels)):
        for j in range(1, len(labels)):
            if i >= j:
                continue
            merged = pd.merge(dfs[i], dfs[j], how='inner', left_on='sequence', right_on='sequence')
            merged_df_pairs.append(merged)

    # Get sequences that appear in more than 1 dataset
    all_seqs_lists = [x["sequence"].tolist() for x in merged_df_pairs]
    seqs_to_query = list(set([j for i in all_seqs_lists for j in i]))

    threads = len(labels)

    if threads_per_task > 1:
        queries_per_task = math.ceil(len(seqs_to_query)/threads_per_task)
        split_query = [seqs_to_query[i*queries_per_task:(i+1)*queries_per_task] for i in range(threads_per_task)]
        all_seq_queries = split_query * len(labels)
        ndfs = []
        for i in range(threads):
            for j in range(threads_per_task):
                ndfs.append(dfs[i])

        p = Pool(threads * threads_per_task)
        start = time.time()
        results = p.starmap(query_seq_in_dataframe, zip(all_seq_queries, ndfs))
        end = time.time()

        print("Process Time", end - start)

        copynums = np.empty((len(seqs_to_query), len(labels)))
        for i in range(threads):
            single_df_results = results[i*threads_per_task:(i+1)*threads_per_task]
            copynums[:, i] = list(chain(*single_df_results))

    else:
        p = Pool(threads)
        start = time.time()
        results = p.starmap(query_seq_in_dataframe, zip(repeat(seqs_to_query), dfs))
        end = time.time()

        print("Process Time", end - start)

        copynums = np.empty((len(seqs_to_query), len(labels)))
        for i in range(threads):
            copynums[:, i] = results[i]

    copynum_dict = {r: copynums[:, rid] for rid, r in enumerate(labels)}
    ndf = pd.DataFrame({"sequence": seqs_to_query, **copynum_dict})
    return ndf


def query_seq_in_dataframe(seqs_to_query, dataframe):
    query_results = []
    dataframe_seqs = list(dataframe["sequence"].values)
    for s in seqs_to_query:
        if s in dataframe_seqs:
            row = dataframe.iloc[dataframe_seqs.index(s)]
            query_results.append(row["copy_num"])
        else:
            query_results.append(np.nan)
    return np.asarray(query_results)


# Prepares data and puts it into a fasta file
# target_dir is where the files should be saved to
# master_df is what we get from process_raw_fasta_files
# Character Conversion will replace characters of strings. Must be dict. ex. {"T": "U"} will replace all Ts with Us
# Remove chars deletes sequences with the provided chars. Must be a list of chars
def prepare_data_files(length_indices, gap_indices, master_df, target_dir, character_conversion=None, remove_chars=None):
    # Make directory for files if not already specified
    if not os.path.isdir(target_dir):
        os.mkdir(f"./{target_dir}")

    labels = list(set(master_df["label"].tolist()))
    for label in labels:
        label_data = master_df[master_df["label"] == label]
        r_seqs = label_data.sequence.tolist()
        r_copynum = label_data.copy_num.tolist()
        if remove_chars is not None:
            for char in remove_chars:
                og_len = len(r_seqs)
                rs, rc = zip(*[(seq, copy_num) for seq, copy_num in zip(r_seqs, r_copynum) if seq.find(char) == -1])
                r_seqs, r_copynum = list(rs), list(rc)
                new_len = len(r_seqs)
                print(f"Removed {og_len-new_len} sequences with character {char}")

        if character_conversion is not None:
            for key, value in character_conversion.items():
                r_seqs = [x.replace(key, value) for x in r_seqs]
        extractor(r_seqs, r_copynum, length_indices, outdir=target_dir+label,
                  uniform_length=True, position_indx=gap_indices)
