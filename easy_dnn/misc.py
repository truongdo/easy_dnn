import inspect
import logging
import optparse
import numpy as np


def debug(message):
    "Automatically log the current function details."
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logging.debug("%s: %s in %s:%i" % (
        message,
        func.co_name,
        func.co_filename,
        func.co_firstlineno
    ))


def info(message):
    "Automatically log the current function details."
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logging.info("%s: %s in %s:%i" % (
        message,
        func.co_name,
        func.co_filename,
        func.co_firstlineno
    ))


def error(message):
    "Automatically log the current function details."
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logging.error("%s: %s in %s:%i" % (
        message,
        func.co_name,
        func.co_filename,
        func.co_firstlineno
    ))
    exit(1)


def f_measure(y, t):
    import numpy as np
    from sklearn.metrics import classification_report
    y = np.asarray(y)
    t = np.asarray(t)
    print('')
    print(classification_report(y, t, digits=4))


def inverse_dict(d):
    ivd = {v: k for k, v in d.items()}
    return ivd


def get_opts():
    parser = optparse.OptionParser()
    parser.add_option('--nnet-struct', dest='nnet_struct', default=None, help='NNET structure')
    parser.add_option('--lr', dest='lr', default=0.0001, type=float,
            help='Learning rate (Default 0.0001)')
    parser.add_option('--lr-decay', dest='lr_decay', default=0.8, type=float,
            help='Learning rate decay (Default 0.8). Scale learning rate by lr-decay if improvement is smaller than --start-decay')
    parser.add_option('--start-decay', dest='start_decay', default=0.001, type=float,
            help='Start learning rate decay when improvment is small than this number (Default 0.001)')
    parser.add_option('--lr-stop', dest='lr_stop', default=0.0000001, type=float,
            help='Early stopping when the learning rate falls below this threshold. (Default 0.0000001)')
    parser.add_option('--momentum', dest='momentum', default=None, type=float,
            help='Momentum (Default None)')
    parser.add_option('--gpu', dest='gpu', default=None, type=int,
            help='Use GPU (GPU ID)')
    parser.add_option('--eos-pad', dest='eos_pad', default=None,
            help='Padding (might only meaningful in LSTM training)')
    parser.add_option('--batch-size', dest='batchsize', default=10, type=int,
            help='Batch size (Default 10)')
    parser.add_option('--loss-func', dest='loss_function', default='cross_entropy',
            help='Loss function [cross_entropy|mean_squared_error] (Default cross_entropy)')
    parser.add_option('--random-seed', dest='random_seed', default=None,
            help='Random seed, used to initialized weight (Default None)', type=int)
    parser.add_option('--n-epoch', dest='n_epoch', default=10, type=int,
            help='Maximum number of training epoch (Default 10)')
    parser.add_option('--grad-clip', dest='grad_clip', default=5.0,
            help='Gradient clipping (Default: 5.0)', type=float)
    parser.add_option('--bprop-len', dest='bprop_len', default=1, type=int,
            help='Unchaining len - sentence len - (Default 1)')
    parser.add_option('--forget-on-new-utt', dest='forget_on_new_utt',
            action="store_true",  default=False,
            help='If this is activated, the trainer will forget the history everytime start \
    a new utterance. Please only activate this when use LSTM layer.')
    parser.add_option('--test', dest='fname_test', default=None,
            help='Test data file. Format: fname:in_size(int)-target_size(int)')
    parser.add_option('--decode', dest='fname_decode', default=None,
            help='Decode data file')
    parser.add_option('--dev', dest='fname_dev', default=None,
            help='Development data file. Format: fname:in_size(int)-target_size(int)')
    parser.add_option('--train', dest='fname_train', default=None,
            help='Training data file. Format: fname:in_size(int)-target_size(int)')
    parser.add_option('--in-model', dest='fname_in_model', default=None,
            help='Input model path')
    parser.add_option('--save-model', dest='fname_out_model', default=None,
            help='Save model path')
    return parser.parse_args()


def get_pading(pad_info):
    if pad_info:
        inp_pad, tgt_pad = pad_info.split(';')
        if ',' in inp_pad:
            inp_pad = np.fromstring(inp_pad, dtype=np.float32, sep=',')
        else:
            if inp_pad.isdigit():
                inp_pad = int(inp_pad)
            else:
                inp_pad = float(inp_pad)
        if ',' in tgt_pad:
            tgt_pad = np.fromstring(tgt_pad, dtype=np.float32, sep=',')
        else:
            if tgt_pad.isdigit():
                tgt_pad = int(tgt_pad)
            else:
                tgt_pad = float(tgt_pad)
        return (inp_pad, tgt_pad)
    else:
        return None


if __name__ == "__main__":
    pad = get_pading(None)
    assert pad is None
    pad = get_pading('1;1')
    assert pad == (1, 1)
    pad = get_pading('1;1,2,3')
    assert pad[0] == 1
    assert (pad[1] == np.array([1, 2, 3])).all()
    pad = get_pading('1,2,3;4')
    assert (pad[0] == np.array([1, 2, 3])).all()
    assert pad[1] == 4
    pad = get_pading('1,2,3;4,5,6')
    assert (pad[0] == np.array([1, 2, 3])).all()
    assert (pad[1] == np.array([4, 5, 6])).all()
