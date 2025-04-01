import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from pyts.image import GramianAngularField
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, \
    recall_score, auc, roc_curve, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standard_scale(data):
    scaler = StandardScaler()
    result = scaler.fit_transform(data.reshape(-1, 1))
    return result.squeeze()


def minmax_scale(data):
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data.reshape(-1, 1))
    return result.squeeze()


def gadf_transform(data, image_size=32, sample_range=(-1, 1), method="difference"):
    gadf = GramianAngularField(image_size, sample_range, method=method)
    result = gadf.fit_transform(data.reshape(1, -1))
    return result.squeeze()


def segment_signal(x, window=30, overlap=None, fs=250, copy=True):
    w = int(window * fs)

    if overlap is None or overlap == 0:
        view = np.lib.stride_tricks.sliding_window_view(x, w)[::w]
    else:
        o = int(overlap * fs)
        view = np.lib.stride_tricks.sliding_window_view(x, w)[::o]

    if copy:
        return view.copy()
    else:
        return view


def plot_confusion_matrix(y_test, Y_pred, normalize="pred", cb=False):
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Convert validation observations to one hot vectors
    Y_true = y_test

    # Create confusion matrix and normalizes it over predicted (columns)
    cm = confusion_matrix(Y_true, Y_pred_classes, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    if ~cb:
        disp.im_.colorbar.remove()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def plot_confusion_matrix_multi(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def marginLoss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1

    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def print_stats(y_test, Y_pred):
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Convert validation observations to one hot vectors
    Y_true = y_test

    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    print("Accuracy =", accuracy_score(Y_true, Y_pred_classes))
    print("F1 Score =", f1_score(Y_true, Y_pred_classes, average="macro"))
    print("TPR =", precision_score(Y_true, Y_pred_classes, average="macro"))
    # print("Recall =", recall_score(Y_true, Y_pred_classes, average="macro"))
    print("TNR = ", recall_score(np.logical_not(Y_true), np.logical_not(Y_pred_classes), average="macro"))
    print("AUC =", roc_auc)


def plot_roc(y_test, Y_pred, name):
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(y_test, axis=1)

    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    rocdisp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    rocdisp.plot()
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def plot_history(history):
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history["Efficient_CapsNet_loss"], color="b", label="Training loss")
    ax[0].plot(history.history["val_Efficient_CapsNet_loss"], color="r", label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc="best")

    ax[1].plot(history.history["Efficient_CapsNet_accuracy"], color='b', label="Training accuracy")
    ax[1].plot(history.history["val_Efficient_CapsNet_accuracy"], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc="best")


def get_lr_callback(batch_size=8, mode='cos', epochs=10):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == 'exp':
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step':
            lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = np.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + np.cos(phase)) + lr_min
        return lr

    return LearningRateScheduler(lrfn, verbose=False)


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x


def calculate_smape(actual, predicted) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)

    return round(
        np.mean(
            np.abs(predicted - actual) /
            (np.abs(predicted) + np.abs(actual))
        )
    )

def butter_bandpass(data, low, high, fs, order=2):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    filtered = filtfilt(b, a, data)
    return filtered
