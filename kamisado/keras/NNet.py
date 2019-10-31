import os
import numpy as np
import tensorflow as tf
from utils import dotdict
from NeuralNet import NeuralNet

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard

from .KamisadoNNet import KamisadoNNet as onnet

USE_CUDA = True
if not USE_CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

args = dotdict({
    'lr': 0.003,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 256,
    'cuda': USE_CUDA,
    'num_channels': 512,
})


class EarlyStoppingEx(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 over_value=0.0001
                 ):
        super(EarlyStoppingEx, self).__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.over_value = over_value

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if not self.monitor_op(current, self.over_value):
            self.wait = 0
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        #self.graph = tf.get_default_graph()
        self.game = game
        self.nnet = onnet(game, args)
        self.board_x, self.board_y, num_encoders = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples, iteration=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        input_boards = self.game.encode_multiple(input_boards)
        #es = EarlyStoppingEx(monitor='v_acc', verbose=1, patience=3, restore_best_weights=True, over_value=0.25,
        #                     min_delta=0.05)
        tb = TensorBoard(log_dir="./logs/iter{}".format(iteration if iteration is not None else ""))
        es = EarlyStopping(monitor='loss', verbose=1, patience=3, restore_best_weights=True)
        checkpoint = ModelCheckpoint(os.path.join('./temp', 'checkpoint.temp.pth.tar'),
                                     monitor='v_acc', verbose=0, save_weights_only=True)
        callbacks_list = [tb, es, checkpoint]
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size,
                            epochs=args.epochs, verbose=1, callbacks=callbacks_list)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = self.game.encode(board)
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
