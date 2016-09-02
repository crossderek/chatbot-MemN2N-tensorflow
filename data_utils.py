from __future__ import absolute_import

import os
import re
import numpy as np

def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 21
    candidates=[]
    with open(os.path.join(data_dir,'dialog-babi-candidates.txt')) as f:
        for line in f:
            line=tokenize(line.lower().strip())[1:]
            # if line[0]=='api_call':
            candidates.append(line)
    return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    if sent=='<silence>':
        return [sent]
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_dialogs(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=str.lower(line.strip())
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            u, r = line.split('\t')
            u = tokenize(u)
            if u[-1] == '.' or u[-1] == '?':
                u = u[:-1]
            r = tokenize(r)
            if r[-1] == '.' or r[-1] == '?':
                r = r[:-1]
            # temporal encoding, and utterance/response encoding
            u.append('$u')
            u.append('#'+str(nid))
            r.append('$r')
            r.append('#'+str(nid))
            context.append(u)
            context.append(r)
        else:
            context=[x for x in context[:-2] if x]
            u=u[:-2]
            r=r[:-2]
            key=' '.join(r)
            if key in candid_dic:
                r=candid_dic[key]
                data.append((context, u,  r))
            context=[]
    return data

def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=str.lower(line.strip())
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            u, r = line.split('\t')
            u = tokenize(u)
            if u[-1] == '.' or u[-1] == '?':
                u = u[:-1]
            r = tokenize(r)
            if r[-1] == '.' or r[-1] == '?':
                r = r[:-1]
            # temporal encoding, and utterance/response encoding
            data.append((context,u,candid_dic[' '.join(r)]))
            u.append('$u')
            u.append('#'+str(nid))
            r.append('$r')
            r.append('#'+str(nid))
            context.append(u)
            context.append(r)
        else:
            # clear context
            context=[]
    return data



def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates(candidates,word_idx):
    vec=np.zeros((len(candidates),len(word_idx)+1))
    for i,candidate in enumerate(candidates):
        for w in candidate:
            vec[i][word_idx[w]]=1
    return vec

def vectorize_data(data, word_idx, sentence_size, memory_size, candidates_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(candidates_size)
        y[answer]=1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
