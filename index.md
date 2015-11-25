---
layout: default
---

# Easy DNN #
This is the project page for the __Easy DNN__ toolkit based on [Chainer](http://www.chainer.org).

As it name, the project is written in the way that it is easy to use. For examle,

   ```
   train_dnn.py --nnet-struct="linear(248-1024):lstm(1024-3)"
   ```

will setup a neural network with the structure that is easy to understand without any further explanation! I believe so!

## Why should I use Easy DNN ##
There are a lot of features inside __Easy DNN__, here are some stand out points:

* Defining a nueral net structure easily with many type of layers (Linear, LSTM, ...).
* Supporting both GPU and CPU computation.
* No configuration files, everything is done on command line level.

## Installation ##
1. __[Required]__ Install [Chainer](http://chainer.org), please read [Chainer](http://chainer.org) for GPU support,
<br>
```
pip install chainer
```
<br>
2. [Optional] If you wish to run classification task and reporting F-score, precision output, install __sklearn__ tookit,
<br>
```
pip install git+https://github.com/scikit-learn/scikit-learn.git
```
3. [Optional] If you wish to use an existing w2vec model that is the output from w2vec toolkit, install __smartencoding__,
<br>
  ```
  pip install smartencoding
  ```

## Examples ##
Examples can be found under [easy_dnn/examples](https://github.com/truongdq/easy_dnn/tree/master/examples)

## Documentation ##
* [Data format](http://truongdq.com/easy_dnn/data_format.html)
* [Mini-batch training](http://truongdq.com/easy_dnn/minibatch.html)
* [Learning rate](http://truongdq.com/easy_dnn/learning_rate.html)

## Contact

- At work: do.truong.dj3 [at] is.naist.jp
- Personal: do.q.truong [at] gmail.com

