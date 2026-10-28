
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Reproducibility and CPU training
tf.random.set_seed(42)
np.random.seed(42)
tf.config.set_visible_devices([], "GPU")

# Project paths
os.makedirs("data", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("TensorFlow:", tf.__version__)
print("Training on CPU")

# Load Fashion-MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize images to [-1, 1]
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 30
NUM_EXAMPLES = 16

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(10000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(1)
)

# Generator: noise -> 28x28 grayscale image
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(7 * 7 * 128, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(
            64, 4, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(
            1, 4, strides=2, padding="same", activation="tanh"
        )
    ], name="Generator")
    return model

# Discriminator: image -> real/fake score
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ], name="Discriminator")
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(
        g_loss, generator.trainable_variables
    )
    disc_gradients = disc_tape.gradient(
        d_loss, discriminator.trainable_variables
    )

    gen_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )

    return g_loss, d_loss

# Fixed noise lets us compare generated images across epochs
fixed_noise = tf.random.normal([NUM_EXAMPLES, LATENT_DIM])
gen_history = []
disc_history = []

def save_samples(epoch):
    predictions = generator(fixed_noise, training=False)
    predictions = (predictions + 1.0) / 2.0

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(predictions[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"Generated Fashion Images - Epoch {epoch}")
    plt.tight_layout()
    fig.savefig(f"generated_images/epoch_{epoch:03d}.png")
    plt.close(fig)

# Training loop
for epoch in range(1, EPOCHS + 1):
    g_total = 0.0
    d_total = 0.0
    batches = 0

    for real_images in dataset:
        g_loss, d_loss = train_step(real_images)
        g_total += float(g_loss)
        d_total += float(d_loss)
        batches += 1

    g_avg = g_total / batches
    d_avg = d_total / batches

    gen_history.append(g_avg)
    disc_history.append(d_avg)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Generator Loss: {g_avg:.4f} | "
        f"Discriminator Loss: {d_avg:.4f}"
    )

    save_samples(epoch)
    generator.save("models/generator.keras")
    discriminator.save("models/discriminator.keras")

# Plot training losses
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), gen_history, label="Generator Loss")
plt.plot(range(1, EPOCHS + 1), disc_history, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCGAN Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curves.png")
plt.close()

print("\nTraining complete!")
print("Generated images: generated_images/")
print("Loss plot: results/loss_curves.png")
print("Saved models: models/")
