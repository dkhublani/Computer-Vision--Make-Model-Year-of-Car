# code for ELM doors model


import numpy as np
import tensorflow as tf
import tqdm


# the final layer
from elm import ELM


def softmax(a):
    c = np.max(a, axis=-1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1)
    return exp_a / sum_exp_a


def main():
    # instantiation

    n_input_nodes = 768432
    n_hidden_nodes =  10000
    n_output_nodes = 2

    elm = ELM(
        n_input_nodes=n_input_nodes,
        n_hidden_nodes=n_hidden_nodes,
        n_output_nodes=n_output_nodes,

        loss='mean_squared_error',
        # default = 'mean_squared_error'
        # more options = 'mean _absolute error', 'categorical_crossentropy', and 'binary_crossentropy'
        activation='sigmoid',

    )
    # preparation of dataset

    n_classes = n_output_nodes

    # load dataset
    (x_train, t_train), (x_test, t_test) = 
    # normalise images' values within [0, 1]
    x_train = x_train.reshape(-1, n_input_nodes) / 255.
    x_test = x_test.reshape(-1, n_input_nodes) / 255.
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # convert label data into one-hot-vector format data.
    t_train = to_categorical(t_train, num_classes=n_classes)
    t_test = to_categorical(t_test, num_classes=n_classes)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)

    # divide the training dataset into two datasets:
    # (1) for the initial training phase
    # (2) for the sequential training phase
    # NOTE: the number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # for example, assign int(1.5 * n_hidden_nodes) training samples
    # for the initial training phase.
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]

    # training

    # initial
    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    # sequential
    pbar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i:i + batch_size]
        t_batch = t_train_seq[i:i + batch_size]
        elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # prediction

    # sample 1000 validation samples from x_test
    n = 1000
    x = x_test[:n]
    t = t_test[:n]

    # 'predict' method returns raw values of output nodes.
    y = elm.predict(x)
    # apply softmax function to the output values.
    y = softmax(y)
    print(y)
    # check the answers.
    for i in range(n):
        max_ind = np.argmax(y[i])
        print('========== sample index %d ==========' % i)
        print('estimated answer: class %d' % max_ind)
        print('estimated probability: %.3f' % y[i, max_ind])
        print('true answer: class %d' % np.argmax(t[i]))

    # evaluation

    # loss = elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

    # save model (to save weights)

    print('saving model parameters...')
    elm.save('./checkpoint/model.ckpt')

    # initialize weights of os_elm
    elm.initialize_variables()

    # load model

    # architecture of the model must be same as the one when the weights were saved

    print('restoring model parameters...')
    elm.restore('./checkpoint/model.ckpt')

    # re-evaluation

    # loss = elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))


if __name__ == '__main__':
    main()