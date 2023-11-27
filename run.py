import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import data
import ViT


# Initialize the Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 2
image_size = 72 
patch_size = 18
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]

X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test
num_classes = 2
input_shape = (150,150,3)

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_test,y_test),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)

    return history

if __name__=='__main__':
    vit_classifier = ViT.create_vit_classifier()
    history = run_experiment(vit_classifier)