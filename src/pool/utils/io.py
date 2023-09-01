import pandas as pd
import math
import time
import pickle
from multiprocessing import Pool

from pool.utils.alphabet import get_alphabet


# Fasta File Methods
def write_fasta(seqs, affs, out):
    o = open(out, 'w')
    for xid, x in enumerate(seqs):
        print('>seq' + str(xid) + '-' + str(affs[xid]), file=o)
        print(x, file=o)
    o.close()


def fasta_read_basic(fasta_file, seq_read_counts=False, drop_duplicates=False):
    o = open(fasta_file)
    titles = []
    seqs = []
    all_chars = []
    for line in o:
        if line.startswith('>'):
            if seq_read_counts:
                titles.append(float(line.rstrip().split('-')[1]))
        else:
            seq = line.rstrip()
            letters = set(list(seq))
            for l in letters:
                if l not in all_chars:
                    all_chars.append(l)
            seqs.append(seq)
    o.close()
    if drop_duplicates:
        all_seqs = pd.DataFrame(seqs).drop_duplicates()
        seqs = all_seqs.values.tolist()
        seqs = [j for i in seqs for j in i]

    if seq_read_counts:
        return seqs, titles
    else:
        return seqs


# Fasta File Reader
def fasta_read(fasta_file, alphabet, threads=1, drop_duplicates=False):
    """
    Parameters
    ----------
    fasta_file: str,
        fasta file name and path
    alphabet: str or dict,
        type of data can be {"dna", "rna", or "protein"} or define it's own {'G': 0, 'Q':1, 'Z':2}
    threads: int, optional,
        number of cpu processes to use to read file
    drop_duplicates: bool,

    """
    o = open(fasta_file)
    all_content = o.readlines()
    o.close()

    line_num = math.floor(len(all_content)/threads)
    # Which lines of file each process should read
    initial_bounds = [line_num*(i+1) for i in range(threads)]
    # initial_bounds = initial_bounds[:-1]
    initial_bounds.insert(0, 0)
    new_bounds = []
    for bound in initial_bounds[:-1]:
        idx = bound
        while not all_content[idx].startswith(">"):
            idx += 1
        new_bounds.append(idx)
    new_bounds.append(len(all_content))

    split_content = (all_content[new_bounds[xid]:new_bounds[xid+1]] for xid, x in enumerate(new_bounds[:-1]))

    p = Pool(threads)

    start = time.time()
    results = p.map(process_lines, split_content)
    end = time.time()

    print("Process Time", end-start)
    all_seqs, all_counts, all_chars = [], [], []
    for i in range(threads):
        all_seqs += results[i][0]
        all_counts += results[i][1]
        for char in results[i][2]:
            if char not in all_chars:
                all_chars.append(char)

    # Sometimes multiple characters mean the same thing, this code checks for that and adjusts q accordingly
    adict = get_alphabet(alphabet)

    char_values = [adict[x] for x in all_chars]
    unique_char_values = list(set(char_values))

    q = len(unique_char_values)

    if drop_duplicates:
        if not all_counts:   # check if counts were found from fasta file
            all_counts = [1 for x in range(len(all_seqs))]
        assert len(all_seqs) == len(all_counts)
        df = pd.DataFrame({"sequence": all_seqs, "copy_num":all_counts})
        ndf = df.drop_duplicates(subset="sequence", keep="first")
        all_seqs = ndf.sequence.tolist()
        all_counts = ndf.copy_num.tolist()

    return all_seqs, all_counts, all_chars, q


def dataframe_to_fasta(df, out, count_key="copy_num", sequence_key='sequence'):
    write_fasta(df[sequence_key].tolist(), df[count_key].tolist(), out)


# Worker for fasta_read
def process_lines(assigned_lines):
    titles, seqs, all_chars = [], [], []

    hdr_indices = []
    for lid, line in enumerate(assigned_lines):
        if line.startswith('>'):
            hdr_indices.append(lid)

    for hid, hdr in enumerate(hdr_indices):

        index = assigned_lines[hdr].find("-")  # first dash
        if index > -1:
            try:
                titles.append(float(assigned_lines[hdr][index+1:].rstrip()))
            except IndexError:
                pass

        if hid == len(hdr_indices) - 1:
            seq = "".join([line.rstrip() for line in assigned_lines[hdr + 1:]])
        else:
            seq = "".join([line.rstrip() for line in assigned_lines[hdr + 1: hdr_indices[hid+1]]])

        seqs.append(seq.upper())

    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, titles, all_chars


def gunter_read(gunterfile):
    o = open(gunterfile)
    seqs, copy_num = [], []
    for line in o:
        c_num, seq = line.split()
        seqs.append(seq.upper())
        copy_num.append(float(c_num))
    o.close()

    all_chars = []
    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, copy_num, all_chars


def csv_read(csv_file, sequence_label="sequence", copy_num_label="copy_num"):
    df = pd.read_csv(csv_file)
    seqs = df[sequence_label].tolist()
    copy_num = df[copy_num_label].tolist()

    all_chars = []
    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, copy_num, all_chars


def load_neighbor_file(neigh_file):
    """ Load pickled neighbor number file (generated by submit_neighbor_job.py)

    Parameters
    ----------
    neigh_file: str,
        full name of neighbor file

    Returns
    -------
    data: list
        list of neighbors from neighbor file
    """
    with open(neigh_file, "rb") as o:
        data = pickle.load(o)
    return data
