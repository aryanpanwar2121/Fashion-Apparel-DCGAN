
# Fashion Apparel Image Synthesizer using DCGAN

## Overview
A deep learning project that generates synthetic fashion apparel images
using a Deep Convolutional Generative Adversarial Network (DCGAN).

## Dataset
Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Model Architecture
- Generator: converts random noise into 28x28 grayscale images.
- Discriminator: distinguishes real apparel images from generated images.

## Project Structure
- `src/train.py` — trains the DCGAN.
- `src/generate.py` — generates new apparel images.
- `generated_images/` — epoch-wise generated samples.
- `results/` — loss curves and generated image grids.
- `models/` — saved Generator and Discriminator.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Generate images:

```bash
python src/generate.py
```

## Limitations
- Images are grayscale and 28x28 pixels.
- Image quality depends on training duration and GAN stability.
- Outputs may contain blurry or unrealistic apparel shapes.
