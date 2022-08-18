import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os

# utility function to create performance plots for part 2
def part2Plots(results, save_dir='', filename='', show_plot=True):
    """plots multiple performance curves from multiple training results and
    saves the resultant plot as a png image

    Arguments:
    ----------

    results: list of dictionary objects, each corresponds to
    the result of a training and should have the following key-value
    items:

        'name': string, indicating the user-defined name of the training

        'loss_curve': list of floats, indicating the loss at each step

        'train_acc_curve': list of floats, indicating the the training accuracy at each step

        'val_acc_curve': list of floats indicating the the validation accuracy at each step

        'test_acc': float, indicating the best test accuracy

        'weights': (not used), 2-D float array, weights of the first hidden layer of the trained MLP

    save_dir: string, indicating the path to directory where the plot image is to be saved

    filename: string, indicating the name of the image file. Note that .png will be automatically
    appended to the filename.

    show_plot: bool, whether the figure is to be shown

    Example:
    --------

    visualizing the results of the training

    # assume the '*_value's are known

    >>> result_x = {'name': name_value, 'loss_curve': loss_curve_value,
                    'train_acc_curve': train_acc_curve_value,
                     'val_acc_curve': val_acc_curve_value,
                     'test_acc': test_acc_value}

    >>> results = [result_1, ..., result_x, ..., result_N]

    >>> part2Plots(results, save_dir=r'some\location\to\save', filename='part2Plots')

    """

    color_list = ['#0000ff', '#ff0000', '#d2691e', '#ff00ff', '#00ff00', '#000000', '#373788']
    style_list = ['-', '--']

    num_results = len(results)

    plot_curve_args = [{'c': color_list[k],
                        'linestyle': style_list[0],
                        'linewidth': 2} for k in range(num_results)]

    plot_point_args = [{'c': color_list[k],
                        'marker': 'o',
                        'markersize': 9,
                        'markerfacecolor':  color_list[k]} for k in range(num_results)]



    font_size = 18

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))


    # training loss
    ax = axes[0, 0]
    ax.set_title('training_loss', loc='left', fontsize=font_size)
    for result, args in zip(results, plot_curve_args):
        ax.plot(np.arange(1, len(result['loss_curve']) + 1), result['loss_curve'], label=result['name'], **args)
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='loss', fontsize=font_size)
        ax.tick_params(labelsize=12)

    # get lines for global legend
    lines = ax.get_lines()


    # training and validation accuracy
    ax = axes[0, 1]
    ax.set_title('train_and_val_accuracies', loc='right', fontsize=font_size)
    for result, args in zip(results, plot_curve_args):
        ax.plot(np.arange(1, len(result['train_acc_curve']) + 1), result['train_acc_curve'], label=result['name'],
                **args)
        args['linestyle'] = style_list[1]
        ax.plot(np.arange(1, len(result['val_acc_curve']) + 1), result['val_acc_curve'], label=result['name'],
                **args)
        args['linestyle'] = style_list[0]
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='acc.', fontsize=font_size)
        ax.tick_params(labelsize=12)

        legend_elements = [Line2D([0], [0], color='k', linestyle=style_list[0], lw=2, label='train.'),
                           Line2D([0], [0], color='k', linestyle=style_list[1], lw=2, label='val.')]

        ax.legend(fontsize=12, loc='best', handles=legend_elements)

    # validation vs training accuracy
    ax = axes[1, 1]
    ax.set_title('validation_vs_training_accuracy', loc='right', fontsize=font_size)
    for result, args in zip(results, plot_curve_args):
        ax.plot(result['train_acc_curve'], result['val_acc_curve'], label=result['name'], **args)
        ax.set_xlabel(xlabel='training', fontsize=font_size)
        ax.set_ylabel(ylabel='validation', fontsize=font_size)
        ax.tick_params(labelsize=12)



    # test vs training accuracy
    ax = axes[1, 0]
    ax.set_title('test_vs_training_accuracy', loc='left', fontsize=font_size)
    for result, args in zip(results, plot_point_args):
        train_acc = result['train_acc_curve'][-1]
        test_acc = result['test_acc']
        ax.plot(train_acc, test_acc, label=result['name'],  **args)
        ax.set_xlabel(xlabel='training', fontsize=font_size)
        ax.set_ylabel(ylabel='test', fontsize=font_size)
        ax.tick_params(labelsize=12)

    # global legend
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))

# utility function to create performance plots for part 3
def part3Plots(results, save_dir='', filename='', show_plot=True):
    """plots multiple performance curves from multiple training results and
    saves the resultant plot as a png image

    Arguments:
    ----------

    results: list of dictionary objects, each corresponds to
    the result of a training and should have the following key-value
    items:

        'name': string, indicating the user-defined name of the training

        'relu_loss_curve': list of floats, indicating the loss at each step

        'sigmoid_loss_curve': list of floats, indicating the loss at each step

        'relu_grad_curve': list of floats, indicating the gradient magnitude at each step

        'sigmoid_grad_curve': list of floats, indicating the gradient magnitude at each step

    save_dir: string, indicating the path to directory where the plot image is to be saved

    filename: string, indicating the name of the image file. Note that .png will be automatically
    appended to the filename.

    show_plot: bool, whether the figure is to be shown

    Example:
    --------

    visualizing the results of the training

    # assume the '*_value's are known

    >>> result_x = {'name': name_value, 'relu_loss_curve': relu_loss_curve_value,
                    'sigmoid_loss_curve': sigmoid_loss_curve_value,
                     'relu_grad_curve': relu_grad_curve_value}

    >>> results = [result_1, ..., result_x, ..., result_N]

    >>> part3Plots(results, save_dir=r'some\location\to\save', filename='part3Plots')

    """

    color_list = ['#0000ff', '#ff0000', '#d2691e', '#ff00ff', '#00ff00', '#000000', '#373788']
    style_list = ['-', '--']

    num_results = len(results)

    relu_curve_args = [{'c': color_list[k],
                        'linestyle': style_list[0],
                        'linewidth': 2} for k in range(num_results)]

    sigmoid_curve_args = [{'c': color_list[k],
                        'linestyle': style_list[1],
                        'linewidth': 2} for k in range(num_results)]



    font_size = 18

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    legend_elements = [Line2D([0], [0], color='k', linestyle=style_list[0], lw=2, label='ReLU'),
                       Line2D([0], [0], color='k', linestyle=style_list[1], lw=2, label='Sigmoid')]

    # training loss
    ax = axes[0]
    ax.set_title('training_losses', loc='left', fontsize=font_size)
    for result, relu_args, sigmoid_args in zip(results, relu_curve_args, sigmoid_curve_args):
        ax.plot(np.arange(1, len(result['relu_loss_curve']) + 1),
                result['relu_loss_curve'], label=result['name'], **relu_args)
        ax.plot(np.arange(1, len(result['sigmoid_loss_curve']) + 1),
                result['sigmoid_loss_curve'], label=result['name'], **sigmoid_args)
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='loss', fontsize=font_size)
        ax.tick_params(labelsize=12)
        ax.legend(loc='upper left', fontsize=12, handles=legend_elements)

    # gradient magnitude
    ax = axes[1]
    ax.set_title('gradient_magnitudes', loc='right', fontsize=font_size)
    for result, relu_args, sigmoid_args in zip(results, relu_curve_args, sigmoid_curve_args):
        ax.plot(np.arange(1, len(result['relu_grad_curve']) + 1),
                result['relu_grad_curve'], label=result['name'], **relu_args)
        ax.plot(np.arange(1, len(result['sigmoid_grad_curve']) + 1),
                result['sigmoid_grad_curve'], label=result['name'], **sigmoid_args)
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='|grad_loss|', fontsize=font_size)
        ax.tick_params(labelsize=12)
        ax.legend(loc='upper right', fontsize=12, handles=legend_elements)

    # global legend
    lines = ax.get_lines()[::2]
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))

# utility function to create performance plots for part 4
def part4Plots(result, save_dir='', filename='', show_plot=True):
    """plots multiple performance curves from multiple training results and
    saves the resultant plot as a png image

    Arguments:
    ----------

    result: dictionary object, each corresponds to
    the result of a training and should have the following key-value
    items:

        'name': string, indicating the user-defined name of the training

        'loss_curve_1': list of floats, indicating the loss with .1 lr at each step

        'loss_curve_01': list of floats, indicating the loss with .01 lr at each step

        'loss_curve_001': list of floats, indicating the loss with .001 lr at each step

        'val_acc_curve_1': list of floats, indicating the val acc with .1 lr at each step

        'val_acc_curve_01': list of floats, indicating the val acc with .01 lr at each step

        'val_acc_curve_001': list of floats, indicating the val acc with .001 lr at each step

    save_dir: string, indicating the path to directory where the plot image is to be saved

    filename: string, indicating the name of the image file. Note that .png will be automatically
    appended to the filename.

    show_plot: bool, whether the figure is to be shown

    Example:
    --------

    visualizing the results of the training

    # assume the '*_value's are known

    >>> result = {'name': name_value, 'loss_curve_1': loss_curve_1_value, ...}

    >>> part4Plots(result, save_dir=r'some\location\to\save', filename='part4Plots')

    """

    if isinstance(result, (list, tuple)):
        result = result[0]

    color_list = ['#0000ff', '#ff0000', '#d2691e', '#ff00ff', '#00ff00', '#000000', '#373788']
    style_list = ['-', '--']

    num_curves = 3

    plot_args = [{'c': color_list[k],
                        'linestyle': style_list[0],
                        'linewidth': 2} for k in range(num_curves)]


    key_suffixes = ['1', '01', '001']


    font_size = 18

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    fig.suptitle('training of <%s> with different learning rates'%result['name'],
                 fontsize=font_size, y=0.025)

    # training loss and validation accuracy
    axes[0].set_title('training_losses', loc='left', fontsize=font_size)
    axes[1].set_title('validation_accuracies', loc='right', fontsize=font_size)
    for key_suffix, plot_args in zip(key_suffixes, plot_args):

        loss_curve = result['loss_curve_' + key_suffix]
        acc_curve = result['val_acc_curve_' + key_suffix]
        label = 'lr=0.%s'%key_suffix

        axes[0].plot(np.arange(1, len(loss_curve) + 1),
                     loss_curve, label=label, **plot_args)
        axes[0].set_xlabel(xlabel='step', fontsize=font_size)
        axes[0].set_ylabel(ylabel='loss', fontsize=font_size)
        axes[0].tick_params(labelsize=12)

        axes[1].plot(np.arange(1, len(acc_curve) + 1),
                     acc_curve, label=label, **plot_args)
        axes[1].set_xlabel(xlabel='step', fontsize=font_size)
        axes[1].set_ylabel(ylabel='accuracy', fontsize=font_size)
        axes[1].tick_params(labelsize=12)


    # global legend
    lines = axes[0].get_lines()
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))

# utility function to visualize learned weights
def visualizeWeights(weights, save_dir, filename='weigths'):
    '''visualizes the weights and saves the visualization as a png image

    Arguments:
    ----------
    weights : numpy array of size 784 x D where D is the number of weights
    save_dir : string, path to directory to save the image
    filename : strint, name of the saved image (.png is to be appended automatically)

    Example:
    --------
    visualizing weights at the input layer of a keras.Model object

    # assume classifier is an instance of keras.Model

    >>> weights = classifier.trainable_weights[0].numpy()

    >>> visualizeWeights(weights, save_dir='some\location\to\save', filename='input_weights')
    '''

    weights = weights.T

    num_weights = weights.shape[-1]

    dim = np.ceil(np.sqrt(num_weights)).astype(int)

    fig, axes = plt.subplots(dim, dim)

    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = weights.min(), weights.max()
    for coef, ax in zip(weights.T, axes[:num_weights].ravel()):
        coef = np.squeeze(coef).T if len(weights.shape) > 2 else coef.reshape(28, 28)
        ax.matshow(coef, cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)

        ax.set_xticks(())
        ax.set_yticks(())

    fig.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))

# utility function to visualize dataset
def visualizeDataset(images, labels, save_dir, filename='dataset', num_samples_per_class=8):

    num_classes = np.max(labels) + 1

    fig, axes = plt.subplots(num_classes, num_samples_per_class)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))

    for r in range(num_classes):
        sample_indcs = np.where(labels == r)[0][:num_samples_per_class]
        for n in range(num_samples_per_class):
            axes[r, n].matshow(images[sample_indcs[n]].reshape(28, 28),
                               cmap=plt.cm.gray)
            axes[r, n].set_xticks(())
            axes[r, n].set_yticks(())

    fig.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))
