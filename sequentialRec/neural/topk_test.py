#coding: utf-8
'''
Original Author: Weiping Song
Contact: songweiping@pku.edu.cn

Modified by: Anthony Alford
Contact: w.anthony.alford@gmail.com


This script counts hits/misses in top-k predictions
'''

import numpy as np
import tensorflow as tf
import argparse
import numpy as np
import sys
import time
import math

from .utils import *
from .model import *
from .eval import Evaluation

parser = argparse.ArgumentParser(description='Sequential or session-based recommendation')
parser.add_argument('--model', type=str, default='tcn', help='sequential model: rnn/tcn/transformer. (default: tcn)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default=20, help='max sequence length (default: 20)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout (default: 0.2)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for Adam (default: 0.001)')
parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--neg_size', type=int, default=1, help='size of negative samples (default: 1)')
parser.add_argument('--worker', type=int, default=10, help='number of sampling workers (default: 10)')
parser.add_argument('--nhid', type=int, default=100, help='number of hidden units (default: 100)')
parser.add_argument('--levels', type=int, default=3, help='# of levels (default: 3)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--loss', type=str, default='ns', help='type of loss: ns/sampled_sm/full_sm (default: ns)')
parser.add_argument('--data', type=str, default='gowalla', help='data set name (default: gowalla)')
parser.add_argument('--log_interval', type=int, default=1e2, help='log interval (default: 1e2)')
parser.add_argument('--eval_interval', type=int, default=1e3, help='eval/test interval (default: 1e3)')
parser.add_argument('--top_k', type=int, default=20, help='eval/test accuracy top values')
parser.add_argument('--log_p', type=float, default=6.0, help='negative log prob for flagging (default: 6)')
parser.add_argument('--num_items', type=float, default=2000, help='hardcoded number of items in corpus')

# ****************************** unique arguments for rnn model. *******************************************************
# None

# ***************************** unique arguemnts for tcn model.
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 100)')

# ****************************** unique arguments for transformer model. *************************************************
parser.add_argument('--num_blocks', type=int, default=3, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=2, help='num_heads')
parser.add_argument('--pos_fixed', type=int, default=0, help='trainable positional embedding usually has better performance')

args = parser.parse_args()
tf.set_random_seed(args.seed)

train_data, val_data, test_data, n_items, n_users = data_generator(args)

# we are overriding this
n_items = args.num_items

max_test_len = 20
test_data_per_step = prepare_eval_test(test_data, batch_size=100, max_test_len=max_test_len)

checkpoint_dir = '_'.join(['save', args.data, args.model, str(args.lr), str(args.l2_reg), str(args.emsize), str(args.dropout)])

print(args)
print ('#Item: ', n_items)
print ('#User: ', n_users)

model = NeuralSeqRecommender(args, n_items, n_users)

lr = args.lr

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0) # only difference

def predict(itemids, sess):
  print('predicting...')
  l = min(len(itemids), max_test_len)
  feed_dict = {model.inp:[itemids[:l-1]], model.dropout: 0}
  prediction = sess.run(model.prediction, feed_dict=feed_dict)

  probs = softmax(prediction[-1])
  for i in range(0, len(probs)):
      if probs[i] > 1e-5:
          print('item: {}, p: {:1.2E}'.format(i, probs[i]))


def predictbatch(batch, sess):
  feed_dict = {model.inp: batch[1], model.dropout: 0.}
  feed_dict[model.pos] = batch[2]
  predictions, n_target = sess.run([model.prediction, model.num_target], feed_dict=feed_dict)

  hit = 0
  flagged = 0
  total = 0

  for i in range(0,len(predictions)):
      row = i // args.seq_len
      col = i % args.seq_len
      if col == args.seq_len - 1:
          total = total + 1
          target = batch[1][row][col]
          probs = softmax(predictions[i])
          target_prob = probs[target]
          top_k = np.flip(np.argsort(probs))[:args.top_k]
          if target in top_k:
              hit = hit + 1
          elif -1.0*np.log(target_prob) > args.log_p:
              flagged = flagged + 1

  return hit, flagged, total

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(sess, '{}/{}'.format(checkpoint_dir, 'model.ckpt'))
        print('Restore model successfully')
    else:
        print('Restore model failed!!!!!')

    hit = 0
    flagged = 0
    total = 0
    for batch in test_data_per_step:
        batchhit, batchflagged, batchtotal = predictbatch(batch, sess)
        hit = hit + batchhit
        flagged = flagged + batchflagged
        total = total + batchtotal

    print('Accuracy {}'.format(1.0* hit/total))
    print('Total Items {}'.format(total))
    print('Hit Items {}'.format(hit))
    print('Flagged Items {}'.format(flagged))

if __name__ == '__main__':
    if not os.path.exists(checkpoint_dir):
        print('Checkpoint directory not found!')
        exit(0)
    main()
