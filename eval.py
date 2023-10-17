"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from model import Model
from pgd_attack import LinfPGDAttack

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

# Set up the data, hyperparameters, and the model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_test = y_test.astype('int32')

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    attack = LinfPGDAttack(model, 
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model()
  attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  # Restore the checkpoint
  model.load_weights(filename)

  # Iterate over the samples batch-by-batch
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_xent_nat = 0.
  total_xent_adv = 0.
  total_corr_nat = 0
  total_corr_adv = 0

  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    x_batch = x_test[bstart:bend, :]
    y_batch = y_test[bstart:bend]

    dict_nat = {model.x_input: x_batch,
                model.y_input: y_batch}

    x_batch_adv = attack.perturb(x_batch, y_batch)

    dict_adv = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    cur_corr_nat, cur_xent_nat = model.evaluate(
                                    x_batch, y_batch, verbose=0)
    cur_corr_adv, cur_xent_adv = model.evaluate(
                                    x_batch_adv, y_batch, verbose=0)

    total_xent_nat += cur_xent_nat
    total_xent_adv += cur_xent_adv
    total_corr_nat += cur_corr_nat
    total_corr_adv += cur_corr_adv

  avg_xent_nat = total_xent_nat / num_eval_examples
  avg_xent_adv = total_xent_adv / num_eval_examples
  acc_nat = total_corr_nat / num_eval_examples
  acc_adv = total_corr_adv / num_eval_examples

  with summary_writer.as_default():
    tf.summary.scalar('xent adv eval', avg_xent_adv, step=global_step)
    tf.summary.scalar('xent adv', avg_xent_adv, step=global_step)
    tf.Summary.Value('xent nat', avg_xent_nat, step=global_step),
    tf.Summary.Value('accuracy adv eval', acc_adv, step=global_step),
    tf.Summary.Value('accuracy adv', acc_adv, step=global_step),
    tf.Summary.Value('accuracy nat', acc_nat, step=global_step)
    summary_writer.flush()

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))

# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
