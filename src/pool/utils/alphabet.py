from copy import copy

def inverse_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

# Globals used for Converting Sequence Strings to Integers
aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
aadict = {aa[k]: k for k in range(len(aa))}
aagraph = copy(aadict)

aadict.update({aa[k].lower():k for k in range(len(aa))})
aadict_inverse = inverse_dict(aadict)

dna = ['A', 'C', 'G', 'T', '-']
dnadict = {dna[k]: k for k in range(len(dna))}
dnagraph = copy(dnadict)

dnadict.update({dna[k].lower(): k for k in range(len(dna))})
dnadict_inverse = inverse_dict(dnadict)

rna = ['A', 'C', 'G', 'U', '-']
rnadict = {rna[k]: k for k in range(len(rna))}
rnagraph = copy(rnadict)

rnadict.update({rna[k].lower(): k for k in range(len(rna))})
rnadict_inverse = inverse_dict(rnadict)

# Deal with wildcard values, all are equivalent
for n in ['*', 'N', '-']:
    dnadict[n] = dnadict['-']
    rnadict[n] = rnadict['-']

# Changing X to be the same value as a gap as it can mean any amino acid
for a in ['X', 'x', '*']:
    aadict[a] = aadict['-']

# aadict['B'] = len(aa)
# aadict['Z'] = len(aa)
# aadict['b'] = len(aa)
# aadict['z'] = -1
# aadict['.'] = -1

letter_to_int_dicts = {"protein": aadict, "dna": dnadict, "rna": rnadict}
int_to_letter_dicts = {"protein": aadict_inverse, "dna": dnadict_inverse, "rna": rnadict_inverse}
graph_dicts = {"protein": aagraph, "dna": dnagraph, "rna": rnagraph}


def get_alphabet(alphabet, inverse=False, clean=False):
    """returns dict that maps a letter of symbol to an integer"""
    if type(alphabet) is str:
        try:
            if inverse:
                return inverse_dict(letter_to_int_dicts[alphabet])
            if clean:
                return graph_dicts[alphabet]
            return letter_to_int_dicts[alphabet]
        except KeyError:
            print("Alphabet {alphabet} not supported!")
            exit(1)
    elif type(alphabet) is dict:
        if inverse:
            return inverse_dict(alphabet)
        if clean:
            return inverse_dict(inverse_dict(alphabet))
        return alphabet
    else:
        print(f"Alphabet of Type {type(alphabet)} is not supported!")
        exit(1)

