import keras
import keras.backend as K
import numpy as np
import tensorflow.compat.v1 as tf
from keras.layers import BatchNormalization, Dropout, Conv2D, TimeDistributed
from keras.layers import Lambda, Flatten, Activation, Dense, Input, ConvLSTM2D
from keras.regularizers import l2

tf.disable_v2_behavior()


def T_get_edge_feature(point_cloud_series, nn_idx, k=5):
    #     """Construct edge feature for each point
    #     Please refer to https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/utils/tf_util.py
    #     Args:
    #     point_cloud_series: (batch_size, time_step, num_points, 1, num_dims)
    #                      or (batch_size, time_step, num_points   , num_dims)
    #     nn_idx: (batch_size, num_points, k)
    #     k: int

    #     Returns:
    #     edge features: (batch_size, time_step, num_points, k, num_dims)
    #     """

    assert len(nn_idx.get_shape().as_list()) == 3
    if point_cloud_series.get_shape().as_list()[-2] == 1:
        point_cloud_series = tf.squeeze(point_cloud_series, -2)

    point_cloud_central = point_cloud_series

    point_cloud_shape = point_cloud_series.get_shape()
    batch_size = tf.shape(point_cloud_series)[0]
    time_step = point_cloud_shape[-3].value
    num_points = point_cloud_shape[-2].value
    num_dims = point_cloud_shape[-1].value

    # Shared graph for all subjects in the batch and all time-frames
    nn_idx = tf.expand_dims(nn_idx, axis=1)

    # Create the shared graph

    # Copy the neighborhood definition for each time step
    nn_idx = tf.tile(nn_idx, [1, time_step, 1, 1])  # https://www.tensorflow.org/api_docs/python/tf/tile
    nn_idx = tf.cast(nn_idx, dtype=tf.int32)

    # Create the shared graph for all batches
    idx_ = tf.range(batch_size * time_step) * num_points
    idx_ = tf.reshape(idx_, [batch_size, time_step, 1, 1])
    idx_ = tf.cast(idx_, dtype=tf.int32)

    point_cloud_flat = tf.reshape(point_cloud_series, [-1, num_dims])
    # point_cloud_neighbors defined by k-NN
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    # Copy the central point data for k times
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, 1, k, 1])

    # For each neighbor, one dimension is x_i as the global features,
    # the difference between neighbors and the central point: x_j - x_i, as the local interaction
    # Therefore, feature * 2
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature


def T_conv_bn_max(edge_feature, kernel=2, activation_fn='relu'):
    #     """TimeDistributed conv with max as aggregation
    #     Args:
    #     edge_feature: (batch_size, time_step, num_points, k, num_dims)
    #     kernel: conv kernel units
    #     activation_fn: non-linear activation
    #
    #     Returns:
    #     conv with max aggregation: (batch_size, time_step, num_points, 1, kernel)

    net = TimeDistributed(Conv2D(kernel, (1, 1)))(edge_feature)
    # net = TimeDistributed(BatchNormalization(axis=-1))(net) # BatchNorm, can be enabled
    if activation_fn is not None:
        net = TimeDistributed(Activation(activation_fn))(net)
    return TimeDistributed(Lambda(lambda x: tf.reduce_max(x, axis=-2, keepdims=True)))(net)


def T_edge_conv(point_cloud_series, graph, kernel=2, activation_fn='relu', k=5):
    #     """TimeDistributed conv with max as aggregation,
    #        wrapper for T_get_edge_feature and T_edge_conv
    #     Args:
    #     point_cloud_series: (batch_size, time_step, num_points, 1, num_dims)
    #                      or (batch_size, time_step, num_points   , num_dims)
    #     graph (FC matrix): (num_points, k)
    #     kernel: conv kernel units
    #     activation_fn: non-linear activation
    #     k: no. of neighbors for cGCN
    #
    #     Returns:
    #     conv output: (batch_size, time_step, num_points, 1, kernel)

    # assert len(graph.get_shape().as_list()) == 2
    graph = Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis=0),
                                     [tf.shape(x[1])[0], 1, 1]))([graph, point_cloud_series])
    edge_feature = Lambda(lambda x: T_get_edge_feature(point_cloud_series=x[0],
                                                       nn_idx=x[1], k=k))([point_cloud_series, graph])

    return T_conv_bn_max(edge_feature, kernel=kernel, activation_fn=activation_fn)


######################## Model description ########################

def build_model(graph_path, ROI_N, frames, kernels=[8, 8, 8, 16, 32, 32], k=5, l2_reg=1e-4, dp=0.5, num_classes=3):
    ############ load static FC matrix ##############
    #print('load graph:', graph_path)
    adj_matrix = np.load(graph_path)
    graph = adj_matrix.argsort(axis=1)[:, ::-1][:, 1:k + 1]

    ############ define model ############
    main_input = Input((frames, ROI_N, 1), name='points')
    static_graph_input = Input(tensor=tf.constant(graph, dtype=tf.int32), name='graph')

    # 5 conv layers

    # 4 stacking conv layers
    net1 = T_edge_conv(main_input, graph=static_graph_input, kernel=kernels[0], k=k)
    net2 = T_edge_conv(net1, graph=static_graph_input, kernel=kernels[1], k=k)
    net3 = T_edge_conv(net2, graph=static_graph_input, kernel=kernels[2], k=k)
    net4 = T_edge_conv(net3, graph=static_graph_input, kernel=kernels[3], k=k)
    net = Lambda(lambda x: tf.concat([x[0],
                                      x[1], x[2], x[3]], axis=-1))([net1, net2, net3, net4])

    # 1 final conv layer with shortcuts from previous conv layers
    net = T_edge_conv(net, graph=static_graph_input, kernel=kernels[4], k=k)

    net = TimeDistributed(Dropout(dp))(net)
    # ConvLSTM2D layer for temporal info, bettern than RNN
    # L2 reg for recurrent parameters for easy convergence
    net = ConvLSTM2D(kernels[5], kernel_size=(1, 1), padding='same',
                     return_sequences=True, recurrent_regularizer=l2(l2_reg))(net)
    net = TimeDistributed(BatchNormalization())(net)
    net = TimeDistributed(Activation('relu'))(net)
    net = TimeDistributed(Flatten())(net)
    net = TimeDistributed(Dropout(dp))(net)

    # Dense layer with softmax activation
    net = TimeDistributed(Dense(num_classes, activation='softmax',
                                kernel_regularizer=l2(l2_reg)))(net)

    output_class = TimeDistributed(Dense(num_classes, activation="softmax", kernel_regularizer=l2(l2_reg)))(net)
    output_level = TimeDistributed(Dense(1, activation="linear", kernel_regularizer=l2(l2_reg)))(net)

    output_class = Lambda(lambda x: K.mean(x, axis=1), name="class")(output_class)
    output_level = Lambda(lambda x: K.mean(x, axis=1), name="level")(output_level)

    model = keras.models.Model(inputs=[main_input, static_graph_input], outputs=[output_class, output_level])

    return model