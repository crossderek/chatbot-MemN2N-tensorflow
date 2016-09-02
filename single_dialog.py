from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates
from sklearn import metrics
from memn2n import MemN2NDialog
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)


candidates,candid_dic = load_candidates(FLAGS.data_dir, FLAGS.task_id)
# task data
train, test, val = load_dialog_task(FLAGS.data_dir, FLAGS.task_id, candid_dic, FLAGS.OOV)
data = train + test + val

vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in data))
vocab |= reduce(lambda x,y: x|y, (set(candidate) for candidate in candidates) )
vocab=sorted(vocab)
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print ("vocab size:",vocab_size)
print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
candidates_vec=vectorize_candidates(candidates,word_idx)
n_cand = candidates_vec.shape[0]
print("Candidate Size", n_cand)

trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size, n_cand)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size, n_cand)
valS, valQ, valA = vectorize_data(val, word_idx, sentence_size, memory_size, n_cand)


print(testS[0])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2NDialog(batch_size, vocab_size, n_cand,sentence_size, memory_size, FLAGS.embedding_size, candidates_vec, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer)
    saver = tf.train.Saver(max_to_keep=50)
    best_validation_accuracy=0
    if FLAGS.train:
        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t = model.batch_fit(s, q, a)
                total_cost += cost_t

            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    pred = model.predict(s, q)
                    train_preds += list(pred)

                val_preds = model.predict(valS, valQ)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                val_acc = metrics.accuracy_score(val_preds, val_labels)

                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')
                if val_acc>best_validation_accuracy:
                    best_validation_accuracy=val_acc
                    saver.save(sess,"task"+str(FLAGS.task_id)+"_"+FLAGS.model_dir+'model.ckpt',global_step=t)
                else:
                    print("early stopping")
                    break
    else:
        ckpt = tf.train.get_checkpoint_state("task"+str(FLAGS.task_id)+"_"+FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        test_preds = model.predict(testS, testQ)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        print("Testing Accuracy:", test_acc)