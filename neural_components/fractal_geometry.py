import tensorflow as tf

from neural_components.custom_ops import sample_min_max_scaling


class OrdinaryLeastSquares(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(OrdinaryLeastSquares, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x):
        scales = 2 ** tf.range(1, self.max_scale + 1,
                               dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)


class LocalSingularityStrength(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(LocalSingularityStrength, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)
        super(LocalSingularityStrength, self).build(input_shape)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)  # this step can be removed if one ensures that x is non-negative
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        return self.bn(alphas, training=training)
