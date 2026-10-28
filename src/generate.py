
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained Generator
generator = tf.keras.models.load_model("models/generator.keras")

# Generate 16 new images from random noise
LATENT_DIM = 100
noise = tf.random.normal([16, LATENT_DIM])
images = generator(noise, training=False)

# Convert pixel values from [-1, 1] to [0, 1]
images = (images + 1.0) / 2.0

os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

fig.suptitle("DCGAN - Newly Generated Fashion Apparel")
plt.tight_layout()
plt.savefig("results/new_fashion_images.png", dpi=150)
plt.close(fig)

print("16 new fashion images generated successfully!")
print("Saved at: results/new_fashion_images.png")
