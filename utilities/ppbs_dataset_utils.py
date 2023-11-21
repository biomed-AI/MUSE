

import numpy as np

def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:2])
                sequence += line_splitted[2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    return list_origins, list_sequences, list_resids, list_labels


def align_labels(labels, pdb_resids,label_resids=None,format='missing'):
    label_length = len(labels)
    sequence_length = len(pdb_resids)
    if label_resids is not None: # Align the labels with the labels found. Safest option.
        if (label_resids.shape[-1] == 2):  # No model index.
            pdb_resids = pdb_resids[:, -2:]  # Remove model index
        elif (label_resids.shape[-1] == 1):  # No model or chain index.
            pdb_resids = pdb_resids[:, -1:]  # Remove model and chain index

        pdb_resids_str = np.array(['_'.join([str(x) for x in y]) for y in pdb_resids])
        label_resids_str = np.array(['_'.join([str(x) for x in y]) for y in label_resids])
        idx_pdb, idx_label = np.nonzero(pdb_resids_str[:, np.newaxis] == label_resids_str[np.newaxis, :])
        if format == 'sparse': # Unaligned labels are assigned category zero.
            aligned_labels = np.zeros( [sequence_length] + list(labels.shape[1:]), dtype=labels.dtype)
        elif format == 'missing': # Unaligned labels are assigned -1/nan category (unknown label, no backpropagation).
            if labels.dtype == np.int32:
                aligned_labels = np.zeros( [sequence_length] + list(labels.shape[1:]), dtype=labels.dtype) -1
            else:
                aligned_labels = np.zeros( [sequence_length] + list(labels.shape[1:]), dtype=labels.dtype) + np.nan
        else:
            raise ValueError('format not supported')
        aligned_labels[idx_pdb] = labels[idx_label]
    else:
        assert label_length == sequence_length, 'Provided size of label array  (%s) does not match sequence length (%s)' % (
            label_length, sequence_length)
        aligned_labels = labels
    return aligned_labels

