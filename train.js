const mnist = require('mnist-data');
const dgemv = require('@stdlib/blas-base-dgemv');
const dger = require('@stdlib/blas-base-dger');
const dscal = require('@stdlib/blas-base-dscal');
const dfill = require('@stdlib/blas-ext-base-dfill');
const floor = require('@stdlib/math-base-special-floor');
const round = require('@stdlib/math-base-special-round');
const max = require('@stdlib/math-base-special-max');
const min = require('@stdlib/math-base-special-min');
const sqrt = require('@stdlib/math-base-special-sqrt');
const cos = require('@stdlib/math-base-special-cos');
const randu = require('@stdlib/random-base-randu');
const exists = require('@stdlib/fs-exists');
const format = require('@stdlib/string-format');
const isArray = require('@stdlib/assert-is-array');
const isNumber = require('@stdlib/assert-is-number');
const NINF = require('@stdlib/constants-float64-ninf');
const fs = require('fs');
const { relu, reluPrime, softmaxInPlace } = require('./shared-math.js');

const TWO_PI = 2.0 * Math.PI;

function printProgressBar(current, total, startTime, width = 30) {
    const percent = min(1.0, max(0.0, current / total));
    const filledLength = round(width * percent);
    const bar = '#'.repeat(filledLength) + '-'.repeat(width - filledLength);

    const elapsed = (Date.now() - startTime) / 1000;
    const rate = current / elapsed;
    const remaining = (total - current) / max(rate, 1e-9);
    const eta = remaining > 0 ? remaining.toFixed(1) + 's' : '0s';

    process.stdout.write(format('\r[%s] %d%% | ETA: %s | %d/%d batches', bar, round(percent * 100), eta, current, total));
}

function preprocess(data) {
    const images = data.images.values;
    const labels = data.labels.values;

    const processed = [];
    for (let i = 0; i < images.length; i++) {
        const flattened = new Float64Array(784);
        let k = 0;
        for (let row = 0; row < 28; row++) {
            for (let col = 0; col < 28; col++) {
                flattened[k++] = images[i][row][col] / 255.0;
            }
        }
        processed.push({ input: flattened, label: labels[i] });
    }
    return processed;
}

function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = floor(randu() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

class MLP {
    constructor(dimensions) {
        if (!isArray(dimensions) || dimensions.length < 2) {
            throw new TypeError('Invalid network dimensions: expected an array with at least 2 layer sizes');
        }

        this.num_layers = dimensions.length;
        this.sizes = dimensions;
        this.weights = [];
        this.biases = [];

        this.nabla_w = [];
        this.nabla_b = [];
        this.activations = [];
        this.zs = [];
        this.delta_buffers = [];

        for (let i = 0; i < dimensions.length - 1; i++) {
            const fan_in = dimensions[i];
            if (!isNumber(fan_in) || fan_in <= 0) {
                throw new TypeError('Invalid network dimensions: layer sizes must be positive numbers');
            }
            const stdDev = sqrt(2.0 / fan_in);

            const w = new Float64Array(dimensions[i + 1] * dimensions[i]);
            for (let j = 0; j < w.length; j++) {
                w[j] = this.gaussianRandom() * stdDev;
            }
            this.weights.push(w);

            const b = new Float64Array(dimensions[i + 1]);
            dfill(b.length, 0.01, b, 1);
            this.biases.push(b);

            this.nabla_w.push(new Float64Array(w.length));
            this.nabla_b.push(new Float64Array(b.length));
            this.delta_buffers.push(new Float64Array(dimensions[i + 1]));
        }

        for (let i = 0; i < dimensions.length; i++) {
            this.activations.push(new Float64Array(dimensions[i]));
            if (i > 0) {
                this.zs.push(new Float64Array(dimensions[i]));
            }
        }
    }

    gaussianRandom() {
        const u = max(randu(), 1e-12);
        const v = randu();
        return sqrt(-2.0 * Math.log(u)) * cos(TWO_PI * v);
    }

    feedforward(input) {
        this.activations[0].set(input);

        for (let l = 0; l < this.num_layers - 1; l++) {
            const m = this.sizes[l + 1];
            const n = this.sizes[l];

            const z = this.zs[l];
            z.set(this.biases[l]);

            dgemv('row-major', 'no-transpose', m, n, 1.0, this.weights[l], n, this.activations[l], 1, 1.0, z, 1);

            const a_next = this.activations[l + 1];
            const isOutputLayer = l === this.num_layers - 2;

            if (isOutputLayer) {
                a_next.set(z);
                softmaxInPlace(a_next);
            } else {
                for (let j = 0; j < z.length; j++) {
                    a_next[j] = relu(z[j]);
                }
            }
        }
    }

    update_mini_batch(mini_batch, learning_rate) {
        let totalBatchLoss = 0;

        for (let l = 0; l < this.num_layers - 1; l++) {
            dfill(this.nabla_b[l].length, 0.0, this.nabla_b[l], 1);
            dfill(this.nabla_w[l].length, 0.0, this.nabla_w[l], 1);
        }

        const target = new Float64Array(10);
        for (const { input, label } of mini_batch) {
            this.feedforward(input);

            const lastIdx = this.num_layers - 2;
            const outputDelta = this.delta_buffers[lastIdx];
            const outputAct = this.activations[lastIdx + 1];

            dfill(target.length, 0.0, target, 1);
            target[label] = 1.0;

            totalBatchLoss += this.cross_entropy_loss(outputAct, label);

            for (let j = 0; j < outputDelta.length; j++) {
                outputDelta[j] = outputAct[j] - target[j];
            }

            for (let l = lastIdx; l >= 0; l--) {
                const currentDelta = this.delta_buffers[l];

                for (let j = 0; j < currentDelta.length; j++) {
                    this.nabla_b[l][j] += currentDelta[j];
                }

                dger(
                    'row-major',
                    currentDelta.length,
                    this.activations[l].length,
                    1.0,
                    currentDelta,
                    1,
                    this.activations[l],
                    1,
                    this.nabla_w[l],
                    this.activations[l].length
                );

                if (l > 0) {
                    const prevDelta = this.delta_buffers[l - 1];
                    dgemv(
                        'row-major',
                        'transpose',
                        this.sizes[l + 1],
                        this.sizes[l],
                        1.0,
                        this.weights[l],
                        this.sizes[l],
                        currentDelta,
                        1,
                        0.0,
                        prevDelta,
                        1
                    );

                    const z_prev = this.zs[l - 1];
                    for (let j = 0; j < prevDelta.length; j++) {
                        prevDelta[j] *= reluPrime(z_prev[j]);
                    }
                }
            }
        }

        const eta = learning_rate / mini_batch.length;
        for (let l = 0; l < this.num_layers - 1; l++) {
            dscal(this.nabla_b[l].length, eta, this.nabla_b[l], 1);
            dscal(this.nabla_w[l].length, eta, this.nabla_w[l], 1);

            for (let j = 0; j < this.biases[l].length; j++) {
                this.biases[l][j] -= this.nabla_b[l][j];
            }
            for (let j = 0; j < this.weights[l].length; j++) {
                this.weights[l][j] -= this.nabla_w[l][j];
            }
        }

        return totalBatchLoss;
    }

    saveModel(filename) {
        const data = {
            sizes: this.sizes,
            weights: this.weights.map((w) => Array.from(w)),
            biases: this.biases.map((b) => Array.from(b))
        };
        fs.writeFileSync(filename, JSON.stringify(data));
        console.log(format('Model saved to %s', filename));
    }

    static loadModel(filename) {
        if (!exists.sync(filename)) {
            throw new Error(format('Model file not found: %s', filename));
        }

        const data = JSON.parse(fs.readFileSync(filename, 'utf-8'));
        if (!isArray(data.sizes) || !isArray(data.weights) || !isArray(data.biases)) {
            throw new Error('Invalid model format: expected sizes, weights, and biases arrays');
        }

        const net = new MLP(data.sizes);
        net.weights = data.weights.map((w) => Float64Array.from(w));
        net.biases = data.biases.map((b) => Float64Array.from(b));

        return net;
    }

    train(trainingData, testData, epochs, batchSize, lr) {
        console.log(format('Training parameters: %d epochs, batch size %d, learning rate %f', epochs, batchSize, lr));
        console.log('---------------------------------------------------------');

        const totalBatches = floor((trainingData.length + batchSize - 1) / batchSize);

        for (let epoch = 0; epoch < epochs; epoch++) {
            const start = Date.now();
            let totalEpochLoss = 0;

            shuffle(trainingData);

            for (let i = 0; i < trainingData.length; i += batchSize) {
                const batch = trainingData.slice(i, i + batchSize);
                totalEpochLoss += this.update_mini_batch(batch, lr);

                const batchNum = floor(i / batchSize) + 1;
                if (batchNum % 10 === 0 || batchNum === totalBatches) {
                    printProgressBar(batchNum, totalBatches, start);
                }
            }

            const avgLoss = totalEpochLoss / trainingData.length;
            const duration = round(((Date.now() - start) / 100)) / 10;
            const accuracy = this.evaluate(testData);

            process.stdout.write('\r' + ' '.repeat(80) + '\r');
            console.log(format('Epoch %d/%d | Loss: %.4f | Accuracy: %.2f%% | Time: %.1fs', epoch + 1, epochs, avgLoss, accuracy, duration));
        }
    }

    evaluate(testData) {
        let correct = 0;
        for (const { input, label } of testData) {
            this.feedforward(input);

            const logits = this.activations[this.num_layers - 1];

            let maxVal = NINF;
            let prediction = -1;

            for (let i = 0; i < logits.length; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    prediction = i;
                }
            }

            if (prediction === label) {
                correct++;
            }
        }
        return (correct / testData.length) * 100;
    }

    cross_entropy_loss(output, label) {
        const epsilon = 1e-15;
        return -Math.log(max(output[label], epsilon));
    }
}

const raw_train = mnist.training(0, 60000);
const raw_test = mnist.testing(0, 10000);

function getArgValue(flag) {
    const idx = process.argv.indexOf(flag);
    if (idx >= 0 && idx + 1 < process.argv.length) {
        return process.argv[idx + 1];
    }
    return undefined;
}

function runTraining() {
    console.log('Preprocessing data...');
    const train_set = preprocess(raw_train);
    const test_set = preprocess(raw_test);

    const net = new MLP([784, 256, 128, 10]);
    console.log('Starting training...');
    net.train(train_set, test_set, 10, 32, 0.1);
    net.saveModel('model.json');
}

function runTestSuite(modelPath) {
    console.log(format('Loading model from %s...', modelPath));
    const net = MLP.loadModel(modelPath);

    console.log('Preprocessing test data...');
    const test_set = preprocess(raw_test);

    const accuracy = net.evaluate(test_set);
    console.log('Test suite completed.');
    console.log(format('Model accuracy on MNIST test set: %.2f%%', accuracy));
}

const shouldTest = process.argv.includes('--test');
const modelPath = getArgValue('--model') || 'model.json';

if (shouldTest) {
    runTestSuite(modelPath);
} else {
    runTraining();
}
