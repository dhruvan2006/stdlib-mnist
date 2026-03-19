import numpy as np
import json
import sys
import time
import math
import os
from sklearn.datasets import fetch_openml

TWO_PI = 2.0 * math.pi

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(np.float64)

def softmax(z):
    max_val = np.max(z)
    exp_z = np.exp(z - max_val)
    return exp_z / np.sum(exp_z)

def cross_entropy_loss(output, label):
    epsilon = 1e-15
    return -np.log(max(output[label], epsilon))

def print_progress_bar(current, total, start_time, width=30):
    percent = min(1.0, max(0.0, current / total))
    filled_length = round(width * percent)
    bar = '#' * filled_length + '-' * (width - filled_length)

    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / max(rate, 1e-9)
    eta = f"{remaining:.1f}s" if remaining > 0 else "0s"

    sys.stdout.write(f"\r[{bar}] {round(percent * 100)}% | ETA: {eta} | {current}/{total} batches")
    sys.stdout.flush()

def preprocess(images, labels):
    processed = []
    for i in range(len(images)):
        flattened = images[i] / 255.0
        processed.append({
            'input': flattened.astype(np.float64),
            'label': int(labels[i])
        })
    return processed

class MLP:
    def __init__(self, dimensions):
        if not isinstance(dimensions, list) or len(dimensions) < 2:
            raise TypeError("Invalid network dimensions: expected a list with at least 2 layer sizes")

        self.num_layers = len(dimensions)
        self.sizes = dimensions
        
        self.weights = []
        self.biases = []

        self.nabla_w = []
        self.nabla_b = []
        self.activations = [np.zeros(dim, dtype=np.float64) for dim in dimensions]
        self.zs = [np.zeros(dim, dtype=np.float64) for dim in dimensions[1:]]
        self.delta_buffers = [np.zeros(dim, dtype=np.float64) for dim in dimensions[1:]]

        for i in range(self.num_layers - 1):
            fan_in = dimensions[i]
            fan_out = dimensions[i + 1]
            if not isinstance(fan_in, int) or fan_in <= 0:
                raise TypeError("Invalid network dimensions: layer sizes must be positive numbers")
            
            std_dev = math.sqrt(2.0 / fan_in)
            
            w = np.random.randn(fan_out, fan_in).astype(np.float64) * std_dev
            self.weights.append(w)
            
            b = np.full(fan_out, 0.01, dtype=np.float64)
            self.biases.append(b)

            self.nabla_w.append(np.zeros_like(w))
            self.nabla_b.append(np.zeros_like(b))

    def feedforward(self, input_data):
        self.activations[0] = np.copy(input_data)

        for l in range(self.num_layers - 1):
            self.zs[l] = np.dot(self.weights[l], self.activations[l]) + self.biases[l]
            
            is_output_layer = (l == self.num_layers - 2)
            if is_output_layer:
                self.activations[l + 1] = softmax(self.zs[l])
            else:
                self.activations[l + 1] = relu(self.zs[l])

    def update_mini_batch(self, mini_batch, learning_rate):
        total_batch_loss = 0.0

        for l in range(self.num_layers - 1):
            self.nabla_b[l].fill(0.0)
            self.nabla_w[l].fill(0.0)

        target = np.zeros(10, dtype=np.float64)

        for sample in mini_batch:
            x, label = sample['input'], sample['label']
            self.feedforward(x)

            last_idx = self.num_layers - 2
            output_act = self.activations[last_idx + 1]

            target.fill(0.0)
            target[label] = 1.0

            total_batch_loss += cross_entropy_loss(output_act, label)

            self.delta_buffers[last_idx] = output_act - target

            for l in range(last_idx, -1, -1):
                current_delta = self.delta_buffers[l]

                self.nabla_b[l] += current_delta
                
                self.nabla_w[l] += np.outer(current_delta, self.activations[l])

                if l > 0:
                    prev_delta = np.dot(self.weights[l].T, current_delta)
                    z_prev = self.zs[l - 1]
                    
                    self.delta_buffers[l - 1] = prev_delta * relu_prime(z_prev)

        eta = learning_rate / len(mini_batch)
        for l in range(self.num_layers - 1):
            self.biases[l] -= self.nabla_b[l] * eta
            self.weights[l] -= self.nabla_w[l] * eta

        return total_batch_loss

    def save_model(self, filename):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        net = MLP(data['sizes'])
        net.weights = [np.array(w, dtype=np.float64) for w in data['weights']]
        net.biases = [np.array(b, dtype=np.float64) for b in data['biases']]
        return net

    def train(self, training_data, test_data, epochs, batch_size, lr):
        print(f"Training parameters: {epochs} epochs, batch size {batch_size}, learning rate {lr}")
        print("-" * 57)

        total_batches = math.floor((len(training_data) + batch_size - 1) / batch_size)

        for epoch in range(epochs):
            start = time.time()
            total_epoch_loss = 0.0

            np.random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                total_epoch_loss += self.update_mini_batch(batch, lr)

                batch_num = math.floor(i / batch_size) + 1
                if batch_num % 10 == 0 or batch_num == total_batches:
                    print_progress_bar(batch_num, total_batches, start)

            avg_loss = total_epoch_loss / len(training_data)
            duration = round((time.time() - start) * 10) / 10
            accuracy = self.evaluate(test_data)

            sys.stdout.write('\r' + ' ' * 80 + '\r')
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Time: {duration:.1f}s")

    def evaluate(self, test_data):
        correct = 0
        for sample in test_data:
            self.feedforward(sample['input'])
            logits = self.activations[-1]
            prediction = np.argmax(logits)
            
            if prediction == sample['label']:
                correct += 1
                
        return (correct / len(test_data)) * 100

def get_arg_value(flag):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None

def fetch_and_prep_data():
    print("Fetching and preprocessing data (this might take a few seconds)...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    
    X = mnist.data
    y = mnist.target
    
    train_set = preprocess(X[:60000], y[:60000])
    test_set = preprocess(X[60000:], y[60000:])
    return train_set, test_set

def run_training():
    train_set, test_set = fetch_and_prep_data()

    net = MLP([784, 256, 128, 10])
    print("Starting training...")
    net.train(train_set, test_set, 10, 32, 0.1)
    net.save_model("model.json")

def run_test_suite(model_path):
    print(f"Loading model from {model_path}...")
    net = MLP.load_model(model_path)

    _, test_set = fetch_and_prep_data()

    accuracy = net.evaluate(test_set)
    print("Test suite completed.")
    print(f"Model accuracy on MNIST test set: {accuracy:.2f}%")

if __name__ == "__main__":
    should_test = '--test' in sys.argv
    model_path = get_arg_value('--model') or 'model.json'

    if should_test:
        run_test_suite(model_path)
    else:
        run_training()

