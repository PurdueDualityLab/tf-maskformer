class SelfAttentionLayer(tf.keras.layers.Layer):

    def __init__(self,
                 dmodel,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 **kwargs):
        self._d_model = dmodel
        self._nhead = nhead
        self._dropout = dropout
        self._activation = activation
        self._normalize_before = normalize_before

        super().__init__(**kwargs)

    def build(self):
        self.self_attn = tf.keras.layers.MultiHeadAttention(
                                num_heads=self._nhead,
                                key_dim=self._d_model,
                                dropout=self._attention_dropout,
                                )
        self.norm = tf.keras.layers.LayerNormalization(
                            axis=-1,
                            dtype=tf.float32)
        self.dropout = tf.keras.layers.Dropout(self._dropout)

        super().build()

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[tf.Tensor] = None,
                     query_pos: Optional[tf.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(query = q, key = k, value=tgt, attention_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[tf.Tensor] = None,
                    query_pos: Optional[tf.Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(query = q, key = k, value=tgt2, attention_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def call(self, tgt,
                tgt_mask: Optional[tf.Tensor] = None,
                query_pos: Optional[tf.Tensor] = None):
        if self._normalize_before:
            return self.forward_pre(tgt, tgt_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, query_pos)

class CrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dmodel,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 **kwargs
                ):
        self._dmodel=dmodel
        self._nhead=nhead
        self._dropout=dropout
        self._activation=activation
        self._normalize_before=normalize_before

        super().__init__(**kwargs)

    def build(self):
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(
                                num_heads=self._nhead,
                                key_dim=self._d_model,
                                dropout=self._attention_dropout,
                                )
        self.norm = tf.keras.layers.LayerNormalization(
                            axis=-1,
                            dtype=tf.float32)
        self.dropout = tf.keras.layers.Dropout(self._dropout)
        super().build()

    def with_pos_embed(self, tensor: Optional[tf.Tensor], pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[tf.Tensor] = None,
                     pos: Optional[tf.Tensor] = None,
                     query_pos: Optional[tf.Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attention_mask=memory_mask,
                                   )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[tf.Tensor] = None,
                    pos: Optional[tf.Tensor] = None,
                    query_pos: Optional[tf.Tensor] = None):

        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def call(self,
                tgt,
                memory,
                memory_mask: Optional[tf.Tensor] = None,
                pos: Optional[tf.Tensor] = None,
                query_pos: Optional[tf.Tensor] = None):
        if self._normalize_before:
            return self.forward_pre(tgt,
                                    memory,
                                    memory_mask,
                                    pos,
                                    query_pos)
        return self.forward_post(tgt,
                                 memory,
                                 memory_mask,
                                 pos,
                                 query_pos)

class FFNLayer(tf.keras.layers.Layer):

    def __init__(self,
                 dmodel,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        self._dmodel=dmodel
        self._dim_feedforward=dim_feedforward
        self._dropout=dropout
        self._activation=activation
        self._normalize_before=normalize_before

        super().__init__()

    def build(self):
        self.linear1 = tf.keras.layers.Dense(self._dim_feedforward)
        self.linear2 = tf.keras.layers.Dense(self._dmodel)
        self.dropout = tf.keras.layers.Dropout(self._dropout)
        self.norm = tf.keras.layers.LayerNormalization(
                            axis=-1,
                            dtype=tf.float32)
        self.activation = tf.keras.layers.Activation(self._activation)

        super().build()

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def call(self, tgt):
        if self._normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MLP(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers
                 ):
        self._input_dim=input_dim
        self._hidden_dim=hidden_dim
        self._output_dim=output_dim
        self._num_layers=num_layers
        super().__init__()

    def build(self, input_shape):
        h = [self._hidden_dim] * (self._num_layers - 1)
        layers = []
        for n, k in zip([self._input_dim] + h, h + [self._output_dim]):
            layers.append(tf.keras.layers.Dense(k))
        self.layers = layers
        self.activation = tf.keras.layers.Activation("relu")
        super().build(input_shape)

    def call(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self._num_layers - 1 else layer(x)
        return x

class MultiScaleMaskedTransformerDecoder(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels = 256,
                 num_classes: int = 133,
                 hidden_dim: int = 256,
                 num_queries: int = 100,
                 nheads: int = 8,
                 dim_feedforward: int = 2048,
                 dec_layers: int = 10,
                 pre_norm: bool = False,
                 mask_dim: int = 256,
                 enforce_input_project: bool = False,
                 ):
        super().__init__()

        self._in_channels=in_channels
        self._num_classes=num_classes
        self._hidden_dim=hidden_dim
        self._num_queries=num_queries
        self._nheads=nheads
        self._dim_feedforward=dim_feedforward
        self._dec_layers=dec_layers
        self._pre_norm=pre_norm
        self._mask_dim=mask_dim
        self._enforce_input_project=enforce_input_project

    def build(self, input_shape):

        self.transformer_self_attention_layers = []
        self.transformer_cross_attention_layers = []
        self.transformer_ffn_layers = []

        for _ in range(self._dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    dmodel=self._hidden_dim,
                    nhead=self._nheads,
                    dropout=0.0,
                    normalize_before=self._pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    dmodel=self._hidden_dim,
                    nhead=self._nheads,
                    dropout=0.0,
                    normalize_before=self._pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    dmodel=self._hidden_dim,
                    dim_feedforward=self._dim_feedforward,
                    dropout=0.0,
                    normalize_before=self._hidden_dim,
                )
            )
        self.decoder_norm = tf.keras.layers.LayerNormalization(
                                axis=-1,
                                dtype=tf.float32)
        # learnable query features
        #self.query_feat = tf.keras.layers.Embedding(self._num_queries, self._hidden_dim)
        self.query_feat = self.add_weight(
                            "query_feat",
                            shape=[self._num_queries, self._hidden_dim],
                            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                            dtype=tf.float32)
        # learnable query p.e.
        #self.query_embed = tf.keras.layers.Embedding(self._num_queries, self._hidden_dim)
        self.query_embed = self.add_weight(
                            "query_embed",
                            shape=[self._num_queries, self._hidden_dim],
                            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                            dtype=tf.float32)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        #self.level_embed = tf.keras.layers.Embedding(self.num_feature_levels, self._hidden_dim)
        self.level_embed = self.add_weight(
                            "level_embed",
                            shape=[self.num_feature_levels, self._hidden_dim],
                            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                            dtype=tf.float32)
        
        self.input_proj = []
        for _ in range(self.num_feature_levels):
            if self._in_channels != self._hidden_dim or self._enforce_input_project:
                self.input_proj.append(tf.keras.layers.Conv2D(filters=self.hidden_dim,
                                      kernel_size=1,
                                      padding='same')
                                      )

            else:
                self.input_proj.append(tf.keras.Sequential())

        self.class_embed = tf.keras.layers.Dense(self._num_classes + 1)
        self.mask_embed = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 3)
        self.mask_embed.build([])

        super().build(input_shape)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        print(f"self.decoder_norm(output) SHAPE: {decoder_output.shape}")
        decoder_output = tf.reshape(decoder_output, [-1, self._num_queries, self._hidden_dim])
        print(f"decoder_output reshape SHAPE: {decoder_output.shape}")
        outputs_class = self.class_embed(decoder_output)
        print(f"self.class_embed(decoder_output) SHAPE: {outputs_class.shape}")
        mask_embed = self.mask_embed(decoder_output)
        print(f"self.mask_embed(decoder_output) SHAPE: {mask_embed.shape}")
        outputs_mask = tf.einsum(
            "bqc,bhwc->bhwq", mask_embed, mask_features)
        print(f"tf.einsum SHAPE: {outputs_mask.shape}")

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        print(attn_mask_target_size)
        attn_mask = tf.image.resize(outputs_mask, attn_mask_target_size, method=tf.image.ResizeMethod.BILINEAR)
        print(f"attn_mask resize SHAPE: {attn_mask.shape}")
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = tf.keras.activations.sigmoid(attn_mask)
        batch_size, height, width, num_channels = attn_mask.shape
        # Reshape the tensor
        attn_mask = tf.reshape(attn_mask, (batch_size, num_channels, height * width))
        print(f"attn_mask flatten SHAPE: {attn_mask.shape}")
        attn_mask = tf.expand_dims(attn_mask, axis=1)
        print(f"attn_mask expand SHAPE: {attn_mask.shape}")
        attn_mask = tf.tile(attn_mask, (1, self._nheads, 1, 1))
        print(f"attn_mask tile SHAPE: {attn_mask.shape}")
        attn_mask = tf.reshape(attn_mask, shape=[self._nheads, self._num_queries ,-1])
        print(f"attn_mask flatten SHAPE: {attn_mask.shape}")
        attn_mask = tf.cast(attn_mask < .5, dtype=tf.bool)

        return outputs_class, outputs_mask, attn_mask
    
    def _generate_image_mask(self, features: tf.Tensor) -> tf.Tensor:
        """Generates image mask from input image."""
        mask = tf.zeros([features.shape[0],features.shape[1],features.shape[2]])
        mask = tf.cast(mask, dtype = bool)
        return mask

    def call(self, x, mask_features, mask = None):

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            print("===================================")
            size_list.append(x[i].shape[-3:-1])
            batch_size = x[i].shape[0]


            features = self._generate_image_mask(x[i])
            pos_embed = position_embedding_sine(
                      features, num_pos_features=self._hidden_dim)
            print(f"pos_embed_sine SHAPE: {pos_embed.shape}")
            pos_embed = tf.reshape(pos_embed, [batch_size, -1, self._hidden_dim])
            print(f"pos_embed reshape SHAPE: {pos_embed.shape}")
            pos_embed = tf.keras.layers.Permute((2,1))(pos_embed)
            print(f"pos_embed permute SHAPE: {pos_embed.shape}")

            pos.append(pos_embed)

            sr = self.input_proj[i](x[i])
            print(f"input_proj{i} SHAPE: {sr.shape}")
            sr = tf.reshape(sr, [batch_size, -1, self._hidden_dim])
            print(f"input_proj reshape SHAPE: {sr.shape}")
            sr = tf.keras.layers.Permute((2,1))(sr)
            print(f"input_proj permute SHAPE: {sr.shape}")

            lvl = self.level_embed[i]
            print(f"level_embed{i} SHAPE: {lvl.shape}")
            lvl = lvl[None, :, None]
            print(f"lvl[None, :, None] SHAPE: {lvl.shape}")

            s = sr+lvl
            print(f"sr+lvl SHAPE: {s.shape}")
            src.append(s)

            # flatten NxCxHxW to HWxNxC
            pp = tf.reshape(pos[-1], [-1, batch_size, self._hidden_dim])
            print(f"Reshape pos[-1] SHAPE: {pp.shape}")
            ps = tf.reshape(src[-1], [-1, batch_size, self._hidden_dim])
            print(f"Reshape src[-1] SHAPE: {ps.shape}")
            pos[-1] = pp
            src[-1] = ps


        _, bs, _ = src[0].shape
        # QxNxC
        #query_embed = tf.keras.layers.tile(tf.expand_dims(self.query_embed.weight, 1), (1, bs, 1))
        #output = tf.keras.layers.tile(tf.expand_dims(self.query_feat.weight, 1), (1, bs, 1))
        qe = self.query_embed
        print(f"query embed SHAPE: {qe.shape}")
        qe = tf.expand_dims(qe, axis=1)
        print(f"Expand dims qe SHAPE: {qe.shape}")
        qe = tf.tile(qe, (1, bs, 1))
        print(f"Tile qe SHAPE: {qe.shape}")
        
        query_embed = qe

        o = self.query_feat
        print(f"query feat SHAPE: {o.shape}")
        o = tf.expand_dims(o, axis=1)
        print(f"Expand dims o SHAPE: {o.shape}")
        o = tf.tile(o, (1, bs, 1))
        print(f"Tile o SHAPE: {o.shape}")
        output = o


        predictions_class = []
        predictions_mask = []

        print("====================================================")
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self._dec_layers):
            level_index = i % self.num_feature_levels
            attn_mask_sum = tf.reduce_sum(tf.cast(attn_mask,tf.float32), axis=-1)
            indices = tf.where(tf.equal(attn_mask_sum, attn_mask.shape[-1]))
            attn_mask = tf.tensor_scatter_nd_update(attn_mask, indices, False)
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                pos=pos[level_index],
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
        }
        
        return out
