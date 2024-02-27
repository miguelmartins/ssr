import tensorflow as tf


class ContractingLayer(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, padding, with_bn=True, **kwargs):
        super(ContractingLayer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.with_bn = with_bn
        self.conv1 = tf.keras.layers.Conv2D(self.n_filters,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            activation=None,
                                            padding=self.padding,
                                            kernel_initializer='HeNormal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.n_filters,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            activation=None,
                                            padding=self.padding,
                                            kernel_initializer='HeNormal')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv1(input_tensor)
        if self.with_bn:
            x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        if self.with_bn:
            x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "with_bn": self.with_bn
        })
        return config


class UpsampleExpandingLayer(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, padding, with_bn=True, **kwargs):
        super(UpsampleExpandingLayer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.with_bn = with_bn
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat_layer = tf.keras.layers.Concatenate()
        self.conv1 = tf.keras.layers.Conv2D(self.n_filters,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            activation=None,
                                            padding=self.padding,
                                            kernel_initializer='HeNormal', )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.n_filters,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            activation=None,
                                            padding=padding,
                                            kernel_initializer='HeNormal', )
        self.bn2 = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "with_bn": self.with_bn
        })
        return config

    def call(self, input_tensor, input_skip_embedding, training=False, **kwargs):
        x = self.upsample(input_tensor)
        x = self.concat_layer([x, input_skip_embedding])

        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        if self.with_bn:
            x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        return x
