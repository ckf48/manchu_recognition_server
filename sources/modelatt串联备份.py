import tensorflow as tf
from keras import backend as K
import keras.layers as KL

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Model:
    # model constants
    batchSize = 32
    imgSize = (120, 400)
    maxTextLen = 30

    def __init__(self, charList, restore=False):
        " initialize model: add CNN, RNN and CTC layers"
        self.charList = charList
        self.restore = restore
        self.ID = 0
        # use normalization over a minibatch
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        # input images
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
        # setup CNN, RNN and CTC layers
        self.CNN_layer()
        self.RNN_layer()
        # self.CTC_layer()
        # setup optimizer to train neural network
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=())
        # 可以看到输出的即为两个batch_normalization中更新mean和variance的操作，需要保证它们在train_op前完成。
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
        self.sess, self.saver = self.setup()

    def CNN_layer(self):
        ''' create CNN layers and reuturn output of these layers '''
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
        # print('CNN输入为%s'%cnnIn4d) (?, 120, 400, 1)
        # list of parameters for the layers
        kernelSizes = [16, 8, 4, 2, 2]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 5), (2, 5), (1, 4), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        pool = cnnIn4d  # input for first CNN layer
        for i in range(numLayers):
            name = 'conv_' + str(i + 1)
            # input: shape = (?, 120, 400 ,1)
            # conv_1: shape = (?, 120, 400, 32)
            # conv_2: shape = (?, 60, 80, 64)
            # conv_3: shape = (?, 30, 16, 128)
            # conv_4: shape = (?, 30, 2, 128)
            # conv_5: shape = (?, 32, 1, 256)
            with tf.variable_scope(name):
                weights_shape = [kernelSizes[i], kernelSizes[i], featureVals[i], featureVals[i + 1]]
                weights = tf.get_variable(name='_weights', shape=weights_shape)
                biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[featureVals[i + 1]]))
                conv = tf.nn.conv2d(input=pool, filter=weights, padding='SAME', strides=(1, 1, 1, 1))
                conv = tf.nn.bias_add(conv, biases, name='pre-activation')
                conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
                # conv_norm = cbam_block(conv_norm)
                relu = tf.nn.relu(conv_norm, name='activation')

            ''' spatial attention 这种方法并行之后才有效
            filter_att_shape = [1, 1, featureVals[i + 1], 1]
            Watt = tf.Variable(tf.truncated_normal(filter_att_shape, stddev=0.1), name="Watt")
            convatt = tf.nn.conv2d(relu, Watt, strides=[1, 1, 1, 1], padding="VALID", name="convatt")
            # convatt = tf.reshape(convatt, [-1, sequence_length - filter_size + 1])
            att_weight = tf.nn.softmax(convatt, name="att_weight_softmax")
            # att_weight = tf.reshape(att_weight, [-1, sequence_length - filter_size + 1, 1, 1])
            att_weight = tf.tile(att_weight, [1, 1, 1, featureVals[i + 1]], name="att_weight_tile")
            attfea = tf.multiply(att_weight, relu)
            '''

            ''' channel attention 
            channels = K.int_shape(relu)[-1]
            x = KL.GlobalAveragePooling2D()(relu)
            x = KL.Reshape((1, 1, channels))(x)
            x = KL.Conv2D(channels, (1, 1), strides=1, name="se_conv1_", padding="valid")(x)
            x = KL.Activation('relu', name='se_conv1_relu_')(x)
            x = KL.Conv2D(channels, (1, 1), strides=1, name="se_conv2_", padding="valid")(x)
            x = KL.Activation('sigmoid', name='se_conv2_relu_')(x)
            output = KL.multiply([relu, x])
            '''

            # pool_1: shape = (?, 60, 80, 32)
            # pool_2: shape = (?, 30, 16, 64)
            # pool_3: shape = (?, 30, 4, 128)
            # pool_4: shape = (?, 30, 2, 128)
            # pool_5: shape = (?, 30, 1, 256)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
            # 中间四个为核大小
        self.cnnOut4d = pool  # shape = (?, 30, 1, 256)

    def RNN_layer(self):
        ''' create RNN layers and return ouput of these layers '''
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])  # shape = (?, 30, 256) 返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
        numHidden = 256
        self.keepprob = tf.placeholder(tf.float32, name='keepprob')
        # basic cells which is used to build RNN
        cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True),
                                               output_keep_prob=self.keepprob) for _ in range(2)]  # 2 layers
        # stack basic cells , 将多个BasicLSTMCell单元汇总为一个
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        ''' rnn attention layers '''
        # stacked = tf.contrib.rnn.AttentionCellWrapper(stacked, attn_length=128)
        # build independent forward and backward bidirectional RNNs
        # return two output sequences fw and bw, shape = (?, 30, 256)
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)
        # concatenate to form a sequence shape = (?, 30, 1, 512) 给定的张量input，该操作插入尺寸索引处的1维axis的input的形状。
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        # project output to chars (including blank), return shape = (?, 30, len(self.charList)+1)
        # tf.truncated_normal(shape, mean, stddev) 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        # 空洞卷积 [filter_height, filter_width, channels, out_channels]
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        # tf.nn.atrous_conv2d out (?, 30 , 1 , len(self.charList)+1)
        # squeeze out (?, 30, len(self.charList)+1)
        '''
        hidden_dim = int(len(self.charList) + 1)

        # hidden_dim 28
        att_Q = tf.Variable(tf.random.truncated_normal(shape=[hidden_dim, hidden_dim]), trainable=True,
                            name='attenion_size_Q')
        att_K = tf.Variable(tf.random.truncated_normal(shape=[hidden_dim, hidden_dim]), trainable=True,
                            name='attenion_size_K')
        att_V = tf.Variable(tf.random.truncated_normal(shape=[hidden_dim, hidden_dim]), trainable=True,
                            name='attenion_size_V')
        outputs = self.rnnOut3d[-1]
        # outputs.shape:(30,28)
        Q = tf.matmul(outputs, att_Q, name='q')
        K = tf.matmul(outputs, att_K, name='k')
        V = tf.matmul(outputs, att_V, name='v')
        qk = tf.matmul(Q, tf.transpose(K, [1, 0], name='t1'), name='qk') / tf.sqrt(
            tf.constant(hidden_dim, dtype=tf.float32, name='scaled_factor'), name='sqrt1')
        weights = tf.nn.softmax(qk)
        self.attout = tf.matmul(weights, V, name='weighted_V')
        # self.attout.shape(30,28)
        # shape = tf.zeros([1, self.attout.shape[0], self.attout.shape[1]])
        self.attout = tf.expand_dims(input=self.attout, axis=0)
        self.out = tf.tile(self.attout, [32, 1, 1])
        '''

    def CTC_layer(self):
        ''' create CTC loss and decoder '''
        # 数组转置，后面的数组代表交换的维度 ctc三个输入形状为[max_time, batch_size, num_classes]
        self.ctcIn3d = tf.transpose(self.rnnOut3d, [1, 0, 2])  # shape = (30, ?, len(self.charList)+1)
        # ground truth texts as sparse tensor (indices, values, dense_shape)
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]),
                                       tf.placeholder(tf.int64, [2]))
        # calculate loss for minibatch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3d, sequence_length=self.seqLen,
                                                  ctc_merge_repeated=True))
        # best path decoder 执行波约束解码
        self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3d, sequence_length=self.seqLen, beam_width=50,
                                                     merge_repeated=False)
        # print(self.decoder)([<tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x0000019F3568E208>], <tf.Tensor 'CTCBeamSearchDecoder:3' shape=(?, 1) dtype=float32>)
        # 返回：
        # 元组(decoded, log_probabilities),其中：top_paths长度的列表,其中decoded[j]是SparseTensor,它包含已解码的输出：
        # decoded[j].indices: Indices matrix (total_decoded_outputs[j] x 2),行存储：[batch, time].
        # decoded[j].values: Values vector, size (total_decoded_outputs[j]),向量存储波束 j 的解码类.
        # decoded[j].dense_shape: Shape vector, size (2),形状值为[batch_size, max_decoded_length[j]]

    def setup(self):
        ''' initialize TensorFlow '''
        sess = tf.Session()
        # saver saves model to file
        saver = tf.train.Saver(max_to_keep=1)
        modelFolder = '../model/'
        latestSaved = tf.train.latest_checkpoint(modelFolder)

        if self.restore and latestSaved:
            # load saved model if available
            saver.restore(sess, latestSaved)
        else:
            sess.run(tf.global_variables_initializer())
        return sess, saver

    def toSparse(self, texts):
        ''' put ground truth texts into sparse tensor for ctc_loss '''
        indices, values = ([] for _ in range(2))
        shape = [len(texts), 0]

        for batchElement, text in enumerate(texts):
            # convert to string of label
            labelStr = [self.charList.index(c) for c in text]
            if len(labelStr) > shape[1]:
                # sparse tensor must have size of maximum
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for i, label in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoderOutputToText(self, ctcOutput, batchSize):
        ''' extract texts from output of CTC decoder '''
        # string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]
        # ctc returns tuple, first element is sparse tensor
        decoded = ctcOutput[0][0]
        # go over all indices and save mapping
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]
            encodedLabelStrs[batchElement].append(label)
        # map labels to chars
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, minibatch):
        ''' training neural network '''
        numBatchElements = len(minibatch.imgs)
        sparse = self.toSparse(minibatch.gtTexts)  # 数据帧转换为SparseDataFrame形式
        # decay learning rate from 0.01 to 0.0001
        lr = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
        feed = {self.inputImgs: minibatch.imgs, self.gtTexts: sparse,
                self.seqLen: [Model.maxTextLen] * numBatchElements, self.learningRate: lr, self.keepprob: 0.75,
                self.is_train: True}
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed)
        self.batchesTrained += 1
        return loss

    def inferBatch(self, minibatch):
        ''' recognize the texts '''
        numBatchElements = len(minibatch.imgs)
        feed = {self.inputImgs: minibatch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements, self.keepprob: 1.0,
                self.is_train: False}
        decoded = self.sess.run(self.decoder, feed_dict=feed)
        texts = self.decoderOutputToText(decoded, numBatchElements)
        return texts

    def inferBatch2(self, minibatch):
        ''' recognize the texts '''
        # numBatchElements = len(minibatch.imgs)
        feed = {self.inputImgs: minibatch.imgs, self.keepprob: 1.0, self.is_train: False}
        rnnOut = self.sess.run(self.rnnOut3d, feed_dict=feed)
        return rnnOut

    def save(self):
        ''' save model to file '''
        self.ID += 1
        self.saver.save(self.sess, '../model/crnn-model', global_step=self.ID)
