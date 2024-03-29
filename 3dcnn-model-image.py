import keras
import tensorflow as tf
model = tf.keras.models.load_model("3dconv_doc_lidc.keras")
tf.keras.utils.plot_model(
    model,
    to_file='3d_model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=False,
    rankdir='TB',
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False
)
