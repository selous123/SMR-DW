from librerank.rl_reranker import *

class SLMR_generator(RLModel):

    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, sample_val=0.2, gamma=0.01,
                 rep_num=1, loss_type='ce'):
        super(SLMR_generator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                                      profile_num, max_norm, sample_val, gamma, rep_num, loss_type)

    def build_ft_chosen(self, data_batch, chosen):
        itm_spar_ph, itm_dens_ph, length_seq = data_batch[2], data_batch[3], data_batch[6]
        batch_size, item_size = len(itm_spar_ph), len(itm_spar_ph[0])
        ret_spar, ret_dens = [], []
        for i in range(batch_size):
            spar_i, dens_i = [], []
            for j in range(item_size):
                if j < length_seq[i]:
                    spar_i.append(itm_spar_ph[i][chosen[i][j]])
                    dens_i.append(itm_dens_ph[i][chosen[i][j]])
                else:
                    spar_i.append(itm_spar_ph[i][length_seq[i]])
                    dens_i.append(itm_dens_ph[i][length_seq[i]])
            ret_spar.append(spar_i)
            ret_dens.append(dens_i)
        return np.array(ret_spar), np.array(ret_dens)

    def attention_based_decoder(self, decoder_inputs, initial_state, attention_states, cell, sampling_function,
                                attention_head_nums=1, feed_context_vector=True, dtype=tf.float32, scope=None):

        # if not decoder_inputs:
        #     raise ValueError("Must provide at least 1 input to attention decoder.")
        if attention_head_nums < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        # TODO: recover
        # if not attention_states.get_shape()[1:2].is_fully_defined():
        #     raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
        #                      % attention_states.get_shape())
        with vs.variable_scope(scope or "point_decoder"):
            batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            input_size = decoder_inputs[0].get_shape()[1].value  # input_size or state_size
            # TODO: recover
            # attn_length = attention_states.get_shape()[1].value # N+1 or N
            attn_length = tf.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value  # state_size——rnn output size

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            # [B,N+1,1,state_size] or [B,N,1,state_size]——encoder outputs
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

            attention_vec_size = attn_size  # state_size——Size of "query" vectors for attention.
            # size is CORRECT! Because both W1 and W2 are square matrix
            hidden_features = []
            v = []
            for a in range(attention_head_nums):
                k = vs.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])  # [1,1,state_size,state_size]
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1],
                                                     "SAME"))  # transformation of encoder outputs, <BOS> in the front of encoder_outputs——W1ej in paper
                v.append(
                    vs.get_variable("AttnV_%d" % a, [attention_vec_size]))  # [state_size]

            states = [initial_state]  # list of all N's decoder state——[B,state_size], may be length of "N+1"

            def attention(query, local_list, i):  # query——[B,state_size], new state produced by decoder in current N
                """Point on hidden using hidden_features(W1ej)——[B,N+1,1,state_size] or [B,N,1,state_size] and query(decoder state)."""
                attention_weights = []  # Results of attention reads will be stored here.
                context_vector_list = []
                dynamic_prefer_vector = self.get_dynamic_vector_dnn(query, local_list, [128, 64, 32, 16, self.num_target],[tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,None], "dynamic_dec_dnn_%d"%i) #[B,dim]->[B, 2]
                for a in range(attention_head_nums):
                    with vs.variable_scope("Attention_%d" % a):
                        y = self.get_hyper_dnn(query, [query.get_shape()[-1].value, 64, attention_vec_size], [tf.nn.relu, None], dynamic_prefer_vector, "hyper_dec_dnn")
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])  # [B,1,1,state_size]
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(
                            v[a] * tf.tanh(hidden_features[a] + y),
                            [2, 3])  # [B,N+1,1,state_size]->[B,N+1] or [B,N,1,state_size]->[B,N]
                        # a = tf.nn.softmax(s)
                        attention_weights.append(s)

                        context_vector = tf.reduce_sum(tf.reshape(s, [-1, attn_length, 1, 1]) * hidden,
                                                       [1, 2])
                        context_vector = tf.reshape(context_vector, [-1, attn_size])
                        context_vector_list.append(context_vector)
                return attention_weights, context_vector_list, dynamic_prefer_vector

            outputs = []  # outputs: list of [B,N+1] or [B,N], may be length of "N+1", attention weight(ACTUAL OUTPUT) of each N's decoder state and all encoder output
            prev = None  # sampled vector
            batch_attn_size = tf.stack([batch_size, attn_size])
            attns = [
                tf.zeros(
                    batch_attn_size, dtype=dtype) for _ in range(attention_head_nums)
            ]  # [B,state_size]

            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])
            inps = []  # list of [B,input_size], decoder inputs, may be length of "N"(except for the first N) or "0"(it depends on "feed_prev")
            prediction_score = tf.zeros([batch_size, attn_length])  # [B,N]
            prefer_vectors = []
            for i, inp in enumerate(decoder_inputs):  # [(N+1)*[B,input_size]], input into decoder N by N
                if i > 0:
                    vs.get_variable_scope().reuse_variables()

                # If sampling_function is set, we use it instead of decoder_inputs.
                if sampling_function is not None and prev is not None:
                    # TODO:reuse=True
                    with vs.variable_scope("sampling_function", reuse=tf.AUTO_REUSE):
                        inp, sampling_symbol_score = sampling_function(prev, i)
                        inps.append(inp) #被挑选商品对应的embedding
                        prediction_score += sampling_symbol_score  # [B,N]
                        # self.dd.append(sampling_symbol_score)

                # Merge input and previous attentions into one vector of the right size.
                # projection dimension should be cell.input_size(input_size), but not cell.output_size (because it should be same with encoder)
                x = inp #[B, dim]
                if feed_context_vector:
                    x = core_rnn_cell._linear([inp] + attns, input_size,
                                              True)  # [B,input_size]——union of input(origin decoder input[B,input_size] or weighted sum of decoder input[B,input_size]) and attns, finally, decoder input of each N is [B,input_size] nor [B,state_size]
                # Run the RNN.
                cell_output, new_state = cell(x, states[-1])  # [B,state_size], [B,state_size]
                states.append(new_state)
                # Run the attention mechanism.
                # TODO: attns should be weighted-sum of attention_states depends on new_state
                # TODO: and output should be linear combination of cell_output and new attns, and prev should be set as output, and then generate new inp?(if sampling_function is not none)
                output, attns, prefer_vector = attention(new_state, inps, i)  # ([B,N+1] or [B,N]) * attention_head_nums, attention information of new decoder state and all encoder output
                prefer_vectors.append(prefer_vector)
                output = tf.stack(output, axis=1)  # [B,attention_head_nums,N]
                output = tf.reduce_mean(output, axis=1)  # [B,N]

                if sampling_function is not None:
                    prev = output
                # The output of the pointer network is actually the attention weight!
                outputs.append(output)
            self.prefer_vector = tf.stack(prefer_vectors, axis=1) #[B, T+1, num_class]
            # print(self.prefer_vector)
            # exit(0)
        return outputs, states, prediction_score

    def deep_set_encode(self):
        
        #初始化以输入的控制参数为核心
        self.enc_input = self.all_feature_concatenation #[B, T, N]
        self.encoder_states = self.get_dnn(self.enc_input, [200], [tf.nn.relu], "enc_dnn_1")  # [B*N or B, N, 200]
        #self.encoder_states = self.get_hyper_dnn(self.enc_input, [self.enc_input.get_shape()[-1].value, 200],[tf.nn.relu],self.init_prefer_vector, "hyper_enc_dnn_1")  # [B*N or B, N, 200]
        final_state = tf.reduce_sum(self.encoder_states, axis=1)  # [B*N or B, 1, 200]
        #self.final_state = self.get_dnn(final_state, [self.lstm_hidden_units], [tf.nn.relu], "enc_dnn_2")
        self.final_state = self.get_hyper_dnn(final_state, [final_state.get_shape()[-1].value, self.lstm_hidden_units], [tf.nn.relu], self.init_prefer_vector, "hyper_enc_dnn")  # [B*N or B, 1, 200]

    def rnn_decode(self):
        # build decoder input
        # training
        self.decoder_inputs = self.build_decoder_input() #[[B,dim]*(N+1)]

        # build sampling function
        training_sampling_function = self.get_training_sampling_function()
        sampling_function = self.get_sampling_function()

        ## 
        with tf.variable_scope("decoder",
                               # partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            training_attention_distribution, states, _ = self.attention_based_decoder(
                self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
                sampling_function=training_sampling_function, attention_head_nums=self.attention_head_nums,
                feed_context_vector=self.feed_context_vector)

        with tf.variable_scope("decoder",
                               # partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                               reuse=True):
            inference_attention_distribution, _, prediction_score = self.attention_based_decoder(
                self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
                sampling_function=sampling_function, attention_head_nums=self.attention_head_nums,
                feed_context_vector=self.feed_context_vector)

        self.training_attention_distribution = training_attention_distribution
        self.training_prediction_order = tf.stack(self.training_prediction_order, axis=1)  # [B,N]
        self.inference_attention_distribution = inference_attention_distribution
        self.predictions = prediction_score  # [B,N]

        self.act_idx_out = self.training_prediction_order
        tmp_idx_out = tf.cast(tf.reshape(self.act_idx_out, [-1, self.item_size, 1]), dtype=tf.int32)
        tmp_idx_range = tf.tile(tf.reshape(tf.range(0, tf.shape(tmp_idx_out)[0]), [-1, 1, 1]),
                                [1, self.item_size, 1])
        tmp_idx_range = tf.cast(tf.concat([tmp_idx_range, tmp_idx_out], axis=2), dtype=tf.int32)
        self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2) #[B, N]
        self.cate_chosen = tf.gather_nd(self.cate_seq, tmp_idx_range)
        # self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2)
        mask = tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N), tf.int32)  # [B,N]
        self.cate_chosen = self.cate_chosen * mask

    def build_decoder_input(self):
        # decoder_inputs = tf.zeros_like(self.enc_input)
        decoder_inputs = [tf.zeros([self.batch_size, self.enc_input.shape[-1].value])] * (
                self.pv_size + 1)  # [[B,input_size]*(N+1)]
        return decoder_inputs

    def symbol_to_index_pair(self, index_matrix):
        # [[3,1,2], [2,3,1]] -> [[[0 3] [0 1] [0 2]],
        #                        [[1 2] [1 3] [1 1]]]
        replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
        rank = len(index_matrix.get_shape())
        if rank == 2:
            replicated_first_indices = tf.tile(
                tf.expand_dims(replicated_first_indices, dim=1),
                [1, tf.shape(index_matrix)[1]])
        return tf.stack([replicated_first_indices, index_matrix], axis=rank)

    def get_sampling_function(self):
        # self.inference_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        self.inference_sampled_symbol = 1 - tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N),
                                                    tf.float32)  # [B,N]
        self.inference_prediction_order = []
        self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights
            if self.use_masking:
                attention_weights = tf.where(self.inference_sampled_symbol > 0, self.neg_inf,
                                             attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                greedy_result = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.inference_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                sampling_symbol_onehot = tf.where(self.inference_sampled_symbol > 0,
                                                  tf.zeros_like(sampling_symbol_onehot),
                                                  sampling_symbol_onehot)  # [B,N]
                # ***** #
                self.inference_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            embedding_matrix = self.enc_input  # [B,N,input_size]
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def get_training_sampling_function(self):
        # self.training_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        self.training_sampled_symbol = 1 - tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N),
                                                   tf.float32)  # [B,N]
        self.training_prediction_order = []
        self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        # self.print_loss = tf.print("training_sampled_symbol: ", self.training_sampled_symbol, output_stream=sys.stderr)
        # self.dd, self.ee = [], []

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights
            if self.use_masking:
                attention_weights = tf.where(self.training_sampled_symbol > 0, self.neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.training_sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
                sampling_symbol = tf.cond(self.feed_train_order, lambda: tf.transpose(self.train_order)[_ - 1, :],
                                          lambda: sampling_symbol)

            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.training_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                # self.print_loss = tf.print("sampling_symbol", sampling_symbol,
                #                            "\nsampling_symbol_onehot", sampling_symbol_onehot,
                #                            output_stream=sys.stderr)
                sampling_symbol_onehot = tf.where(self.training_sampled_symbol > 0,
                                                  tf.zeros_like(sampling_symbol_onehot),
                                                  sampling_symbol_onehot)  # [B,N]
                # self.print_loss = tf.print("sampling_symbol", sampling_symbol,
                #                            "\nsampling_symbol_onehot", sampling_symbol_onehot,
                #                            output_stream=sys.stderr)
                # ***** #
                self.training_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            # self.dd.append([tf.transpose(self.train_order)[_ - 1, :], sampling_symbol_onehot])
            # self.ee.append(sampling_symbol_score)
            # self.print_loss = tf.print("sampling_symbol_score: ", sampling_symbol_score,
            #                            "\nsampling_symbol: ", sampling_symbol_onehot,
            #                            "\nmask", self.training_sampled_symbol,
            #                            output_stream=sys.stderr)
            embedding_matrix = self.enc_input
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def _build_graph(self):
        self.lstm_hidden_units = 32

        with tf.variable_scope("input"):
            self.num_target = 2
            self.train_phase = self.is_train
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")  # True
            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.div_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            self.auc_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            self.init_prefer_vector = tf.placeholder(tf.float32, [None, self.num_target]) #[B,2]
            self.prefer_label = tf.placeholder(tf.float32, [None, None, self.num_target]) #[B,N,2]
            # self.idx_out_act = tf.placeholder(tf.int32, [None, self.max_time_len])
            self.item_input = self.item_seq
            self.item_label = self.label_ph  # [B, N]
            item_features = self.item_input

            self.item_size = self.max_time_len
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])  # [B*N, N]

            self.itm_enc_input = tf.reshape(item_features, [-1, self.item_size, self.ft_num])  # [B, N, ft_num]
            self.usr_enc_input = tf.reshape(self.usr_seq, [-1, 1, self.profile_num * self.emb_dim])
            self.full_item_spar_fts = self.itm_spar_ph
            self.full_item_dens_fts = self.itm_dens_ph
            self.pv_item_spar_fts = tf.reshape(self.full_item_spar_fts, (-1, self.full_item_spar_fts.shape[-1]))
            self.pv_item_dens_fts = tf.reshape(self.full_item_dens_fts, (-1, self.full_item_dens_fts.shape[-1]))

            self.raw_dec_spar_input = tf.placeholder(tf.float32, [None, self.itm_spar_num])
            self.raw_dec_dens_input = tf.placeholder(tf.float32, [None, self.itm_dens_num])
            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.raw_dec_input = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]),
                 self.itm_dens_ph], axis=-1)
            self.dec_input = self.raw_dec_input
            # self.batch_size = tf.shape(self.dec_input)[0]
            self.batch_size = self.dec_input.get_shape()[0].value
            self.N = self.item_size
            self.use_masking = True
            self.training_sample_manner = 'sample'
            self.sample_manner = 'greedy'
            self.pv_size = self.N
            self.attention_head_nums = 2
            self.feed_context_vector = True
            self.feed_train_order = tf.placeholder(tf.bool)
            self.name = 'SLMR_generator'
            self.train_order = tf.placeholder(tf.int64, [None, self.item_size])

        self.feature_augmentation()

        with tf.variable_scope("encoder"):
            self.deep_set_encode()

        with tf.variable_scope("encoder_state"):
            # self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.lstm_hidden_units)

        with tf.variable_scope("decoder"):
            self.rnn_decode()

        with tf.variable_scope("loss"):
            self._build_loss()

    def _build_loss(self):
        self.gamma = 1
        if self.loss_type == 'ce':
            gamma = 0.3

            reinforce_weight = tf.range(self.pv_size, dtype=tf.float32) #[N]
            reinforce_weight = tf.reshape(reinforce_weight, [-1, 1])  # [N,1]
            reinforce_weight = tf.tile(reinforce_weight, [1, self.pv_size])  # [N,N]
            reinforce_weight = reinforce_weight - tf.transpose(reinforce_weight)  # [N,N]
            reinforce_weight = tf.where(reinforce_weight >= 0, tf.pow(gamma, reinforce_weight),
                                        tf.zeros_like(reinforce_weight))  # [N,N]
            # self.print_loss = tf.print("rw: ", reinforce_weight, output_stream=sys.stderr)

            logits = tf.stack(self.training_attention_distribution[:-1], axis=1)  # [B,10,10]
            labels = tf.one_hot(self.training_prediction_order, self.item_size)  # [B,10,10]

            # 需要根据self.div_label 和 self.auc_label 计算下一个坑位应该选择何种策略
            # 初始化策略: self.controllable_auc_prefer
            # [B, N] weights
            #[B, N+1, num_class]
            prefer_vector_label = self.get_prefer_vector_label(self.prefer_label, self.init_prefer_vector) #[B,N+1,2]
            self.prefer_vector_label = prefer_vector_label

            # # MSE
            # diff = self.prefer_vector[:, :-2, :] - prefer_vector_label[:, 1:-1, :] #[B, N-1, 2]
            # diff_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=[1,2])) ##训练dynamic layer可以通过status预测下一个位置的目标是什么

            # BCE Loss #[B,N-1,2]
            ## self.prefer_vector[:, :, :] softmax之后的数据 -> 输出当前坑位哪个目标完成的最不好？
            ## prefer_vector_label[:, :, :] onehot label -> 坑位目标是哪一个？
            epsilon = 1e-10
            log_prefer = tf.log(self.prefer_vector + epsilon)
            cross_entropy = -tf.reduce_sum(prefer_vector_label * log_prefer, axis=[1,2]) #[B,T+1,2] -> [B]
            self.diff_loss = tf.reduce_mean(cross_entropy)#[B,]->[]

            weighted_div_label = tf.reshape(tf.matmul(self.div_label, reinforce_weight), [-1, 1])  # [B,N]
            ## 取出对应位置的概率
            div_ce = tf.multiply(weighted_div_label, tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                               labels=labels), [-1, 1]))
            div_ce = tf.reshape(div_ce, (-1, self.max_time_len))  # [B, N]

            weighted_auc_label = tf.reshape(tf.matmul(self.auc_label, reinforce_weight), [-1, 1])  # [B,N]
            auc_ce = tf.multiply(weighted_auc_label, tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                               labels=labels), [-1, 1]))
            auc_ce = tf.reshape(auc_ce, (-1, self.max_time_len))  # [B, N]

            # ce = tf.add(tf.multiply(div_ce, 1 - self.controllable_auc_prefer),
            #             tf.multiply(auc_ce, self.controllable_auc_prefer))
            ce = tf.stack([auc_ce, div_ce], axis=-1) * prefer_vector_label[:,:-1,:]  #[B, N, num_class]

            self.div_loss = tf.reduce_mean(tf.reduce_sum(div_ce, axis=1))
            self.auc_loss = tf.reduce_mean(tf.reduce_sum(auc_ce, axis=1))
            self.loss = tf.reduce_mean(tf.reduce_sum(ce, axis=[1,2])) + 5.0 * self.diff_loss
            #self.loss = self.diff_loss
        else:
            raise ValueError('No loss.')

        self.opt()
    
    def get_prefer_vector_label(self, label, init_prefer_vector):
        """
        init_prefer_vector: [B, num_class]
        label: [B, T, num_class]
        """
        init_prefer_vector = tf.expand_dims(init_prefer_vector, axis=1) #[B, 1, num_target]

        #label = tf.math.cumsum(label, axis=1)
        cumsum_label = tf.nn.softmax(label, axis=-1) #[B, T, num_class]
        dynamic_prefer_vector = 1 - cumsum_label #[B, T, num_class]
        ## 如果前T-1时刻的某一项reward提升比较大，则T时刻应该减少该任务投入
        dynamic_prefer_vector_hard = 1 - tf.one_hot(tf.argmax(cumsum_label ,axis=-1), depth=self.num_target) #[B,T,num_class]
        prefer_vector_label = tf.concat([init_prefer_vector, dynamic_prefer_vector], axis=1)
        prefer_vector_label_hard = tf.concat([init_prefer_vector, dynamic_prefer_vector_hard], axis=1)
        return prefer_vector_label_hard #[B,T+1,num_class]

    def get_dynamic_vector_dnn(self, status, local_list, layer_nums, layer_acts, name="dynamic_dnn"):
        """
        在generator中定义网络预测下一个action应该以何为目标
        """
        # input_ft = tf.stop_gradient(status) #不要对状态求导
        if len(local_list) == 0:
            embed = tf.reduce_sum(tf.zeros_like(self.enc_input), axis=1) #[b, 210]
        elif len(local_list) == 1:
            embed = local_list[0]
        else:
            embed = tf.reduce_sum(tf.stack(local_list, axis=1), axis=1) #[b,T,dim]->[b,dim]
        #input_ft = tf.concat([tf.stop_gradient(status), embed], axis=1)#[b, dim*2]
        input_ft = embed
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            input_ft = self.get_dnn(input_ft, layer_nums, layer_acts, name) 
        input_ft = tf.nn.softmax(input_ft, axis=-1) #[b,2]
        #input_ft = tf.one_hot(tf.argmax(input_ft ,axis=-1), depth=self.num_target)#[b,2]
        return input_ft #[]

    def get_hyper_dnn(self, x, layer_nums, layer_acts, dynamic_prefer_vector, name="hyper_dnn"): 
        input_ft = x
        assert len(layer_nums) == len(layer_acts) + 1
        with tf.variable_scope(name):
            for i, layer_act in enumerate(layer_acts):
                input_ft = self.build_hyper_mlp_net_scope(input_ft, layer_nums[i], layer_nums[i + 1], 'layer_%d' % i, dynamic_prefer_vector, layer_act)
        return input_ft

    # self defined
    def build_hyper_mlp_net_scope_dnn(self, inp, inp_last_dim, units, scope_name, dynamic_prefer_vector, activation=tf.nn.relu):
        """
        dynamic_prefer_vector: [?, 2]->[?,dim]
        inp:[b,dim]
        """
        dynamic_prefer_vector = tf.one_hot(tf.argmax(dynamic_prefer_vector, axis=-1), depth=self.num_target)
        with tf.variable_scope(scope_name):
            dynamic_ret = tf.contrib.layers.fully_connected(
                inputs=dynamic_prefer_vector,
                num_outputs=inp_last_dim,
                scope=scope_name + '_prefer',
                activation_fn=None,
                reuse=tf.AUTO_REUSE)
            #input_ft = tf.concat([inp, dynamic_prefer_vector], axis=-1) #把condition连在每一层的输入上
            #input_ft = dynamic_ret + inp
            input_ft = tf.concat([dynamic_ret, inp], axis=-1) #[B, dim]
            ret = tf.contrib.layers.fully_connected(
                inputs=input_ft,
                num_outputs=units,
                scope=scope_name + '_ret',
                activation_fn=activation,
                reuse=tf.AUTO_REUSE)
        return ret 

    #copy from CMR
    def build_hyper_mlp_net_scope(self, inp, inp_last_dim, units, scope_name, dynamic_prefer_vector, activation=tf.nn.relu):
        hyper_w = tf.contrib.layers.fully_connected(
            inputs=dynamic_prefer_vector,
            num_outputs=inp_last_dim,
            scope=scope_name + '_w',
            activation_fn=None,
            reuse=tf.AUTO_REUSE)
        hyper_b = tf.contrib.layers.fully_connected(
            inputs=dynamic_prefer_vector,
            num_outputs=inp_last_dim,
            scope=scope_name + '_b',
            activation_fn=None,
            reuse=tf.AUTO_REUSE)
        ret = tf.add(inp * hyper_w, hyper_b) #[bs, inp_last_dim]<-[bs, ild] * [bs, ild] + [bs, ild]
        ret = tf.contrib.layers.fully_connected(
                inputs=ret,
                num_outputs=units,
                scope=scope_name + '_ret',
                activation_fn=None,
                reuse=tf.AUTO_REUSE)
        if activation:
            ret = activation(ret)
        return ret

    def train(self, batch_data, train_order, auc_rewards, div_rewards, lr, reg_lambda, prefer_vector, keep_prop=0.8, train_prefer=0,prefer_label=None):
        B = auc_rewards.shape[0]
        init_prefer_vector = np.tile(np.array([train_prefer, 1-train_prefer]), [B,1])
        with self.graph.as_default():
            _, total_loss, auc_loss, div_loss, diff_loss,  training_attention_distribution, training_prediction_order, predictions, prefer_vector_label = \
                self.sess.run(
                    [self.train_step, self.loss, self.auc_loss, self.div_loss, self.diff_loss,
                     self.training_attention_distribution, self.training_prediction_order, self.predictions, self.prefer_vector_label],
                    feed_dict={
                        self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                        self.itm_spar_ph: batch_data[2],
                        self.itm_dens_ph: batch_data[3],
                        self.seq_length_ph: batch_data[6],
                        self.auc_label: auc_rewards,
                        self.div_label: div_rewards,
                        self.reg_lambda: reg_lambda,
                        self.lr: lr,
                        self.keep_prob: keep_prop,
                        self.is_train: True,
                        self.feed_train_order: True,
                        self.train_order: train_order,
                        self.init_prefer_vector: init_prefer_vector,
                        self.prefer_label:prefer_label
                    })
            return total_loss, auc_loss, div_loss,diff_loss, prefer_vector_label
        

    def rerank(self, batch_data, keep_prop=0.8, train_prefer=0):
        B = np.array(batch_data[1]).shape[0]
        init_prefer_vector = np.tile(np.array([train_prefer, 1-train_prefer]), [B,1])
        # def rerank(self, batch_data, train_prefer, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen, prefer_vector= \
                self.sess.run(
                    [self.training_attention_distribution, self.training_prediction_order, self.predictions,
                     self.cate_seq, self.cate_chosen, self.prefer_vector],
                    feed_dict={
                        self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                        self.itm_spar_ph: batch_data[2],
                        self.itm_dens_ph: batch_data[3],
                        self.seq_length_ph: batch_data[6],
                        self.is_train: True,
                        self.feed_train_order: False,
                        self.train_order: np.zeros_like(batch_data[4]),
                        self.keep_prob: keep_prop,
                        # self.sample_phase: sample_phase,
                        self.label_ph: batch_data[4],
                        self.init_prefer_vector: init_prefer_vector#[B,num_target]
                    },
                )
        return training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen, prefer_vector

    def eval(self, batch_data, reg_lambda, eval_prefer=0, keep_prob=1, no_print=True):
        B = np.array(batch_data[1]).shape[0]
        init_prefer_vector = np.tile(np.array([eval_prefer, 1-eval_prefer]), [B,1])
        with self.graph.as_default():
            rerank_predict = self.sess.run(self.predictions,
                                           feed_dict={
                                               self.usr_profile: np.reshape(np.array(batch_data[1]),
                                                                            [-1, self.profile_num]),
                                               self.itm_spar_ph: batch_data[2],
                                               self.itm_dens_ph: batch_data[3],
                                               self.seq_length_ph: batch_data[6],
                                               self.is_train: False,
                                               self.sample_phase: False,
                                               self.init_prefer_vector: init_prefer_vector,#[B,num_target]
                                               self.keep_prob: 1})
            return rerank_predict, 0