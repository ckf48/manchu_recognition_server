import tensorflow as tf
from modelatt并联备份2 import Model
from dataloader import MiniBatch

class Model2:
    # model constants
    batchSize = 32
    # imgSize = (120, 400)
    imgSize = (60, 200)
    maxTextLen = 30

    def __init__(self, charList, restore=False):
        " initialize model: add CNN, RNN and CTC layers"
        self.charList = charList
        self.restore = restore
        self.ID = 0
        # use normalization over a minibatch
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        # input images
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, 1 , len(self.charList)+1))
        # setup CNN, RNN and CTC layers
        self.CTC_layer()
        # setup optimizer to train neural network
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=())
        # 可以看到输出的即为两个batch_normalization中更新mean和variance的操作，需要保证它们在train_op前完成。
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
        self.sess, self.saver = self.setup()

    def CTC_layer(self):
        ''' create CTC loss and decoder '''
        # 数组转置，后面的数组代表交换的维度 ctc三个输入形状为[max_time, batch_size, num_classes]
        self.ctcIn3d = tf.transpose(self.inputImgs, [1, 0, 2])  # shape = (30, ?, len(self.charList)+1)
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
        print(len(texts))

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
        '''get  bottletensor'''
        model = Model(self.charList, restore=True)
        self.bottletensor = model.inferBatch2(minibatch)
        print('bottletensor是', self.bottletensor)
        
        ''' training neural network '''
        numBatchElements = len(minibatch.imgs)
        sparse = self.toSparse(minibatch.gtTexts)  # 数据帧转换为SparseDataFrame形式
        # decay learning rate from 0.01 to 0.0001
        lr = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
        feed = {self.inputImgs: self.bottletensor, self.gtTexts: sparse,
                self.seqLen: [Model.maxTextLen] * numBatchElements, self.learningRate: lr,
                self.is_train: True}
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed)
        self.batchesTrained += 1
        return loss

    def inferBatch(self, minibatch):
        ''' recognize the texts '''
        numBatchElements = len(minibatch.imgs)
        feed = {self.inputImgs: minibatch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements,
                self.is_train: False}
        decoded = self.sess.run(self.decoder, feed_dict=feed)

        texts = self.decoderOutputToText(decoded, numBatchElements)
        return texts

    def save(self):
        ''' save model to file '''
        self.ID += 1
        self.saver.save(self.sess, '../model/crnn-model', global_step=self.ID)
