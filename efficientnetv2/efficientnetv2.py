from .effnetv2_model import EffNetV2Model
import tensorflow as tf
import os

def EfficientNetV2(
        model_name='efficientnetv2-s',
        weights=None,
        input_shape=None,
        include_top=True,
        dropout_rate=None,
        pooling=True,
        num_class=1000):
    model = EffNetV2Model(model_name=model_name,include_top=include_top,pooling=pooling)
    if not input_shape:
        size = model._mconfig.eval.isize
        input_shape = (size, size, 3)
    x = tf.keras.Input(input_shape)
    output = model.call(x, training=None)

    if pooling and not include_top and num_class:
        if dropout_rate:
            output = tf.keras.layers.Dropout(dropout_rate)(output)
        output = tf.keras.layers.Dense(num_class)(output)

    model = tf.keras.Model(inputs=x,outputs=output)

    if os.path.exists(weights):
        model.load_weights(weights,by_name=True,skip_mismatch=True)
    else:
        raise ValueError('invalid weights path: {}!'.format(weights))

    return model