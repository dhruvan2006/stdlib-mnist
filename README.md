# stdlib MNIST

**[Try the Live Web UI](https://dhruvan2006.github.io/stdlib-mnist/)**

A custom Multilayer Perceptron (MLP) implementation from scratch to recognize handwritten digits from the MNIST dataset. The project extensively uses [stdlib](https://github.com/stdlib-js/stdlib) for math and linear algebra in JavaScript.

All matrix operations, forward passes and backpropagation steps are powered purely by `@stdlib/blas` and `@stdlib/math` packages.

Achieved a 1.78x speedup using native addon's compared to the pure JS fallback implementation.

![Live Demo](./assets/image.png)

## Live Demo

**[Try the Live Web UI](https://dhruvan2006.github.io/stdlib-mnist/)**

## Installation

```bash
git clone https://github.com/dhruvan2006/stdlib-mnist.git
cd stdlib-mnist
npm install
```

## Usage

**1. Training the model**

To train the neural network from scratch, run the training script. This script preprocesses the MNIST data, initializes an MLP with the architecture `[784, 256, 128, 10]`, and trains it over 10 epochs.

```bash
npm run train
```

The weights are saved to `model.json`

**2. Testing the Model**

To evaluate the model against the 10,000-image MNIST testing set:

```bash
npm run test
```

**3. Running the Web UI**

Use any simple HTTP server. For example: `npx serve .` or `python3 -m http.server`.

## `stdlib`

This project heavily relies on the modularity of `@stdlib`. Key packages include:
- **BLAS**: `@stdlib/blas-base-dgemv`, `@stdlib/blas-base-dger`, `@stdlib/blas-base-dscal`
- **Math**: `@stdlib/math-base-special-exp`, `@stdlib/math-base-special-max`, `@stdlib/constants-float64-ninf`
- **Utils**: `@stdlib/random-base-randu`, `@stdlib/string-format`

Experienced a 1.78x speedup using native addon's compared to the pure JS implementation. Each epoch now takes ~17.4 seconds compared to ~31 seconds previously.

- Using native addons:  
![Optimized training using native addons](./assets/optimized.png)

- Using pure JS fallbacks:  
![Unoptimized pure JS training](./assets/js.png)

## License

This project is open-source and available under the [MIT License](./LICENSE).
