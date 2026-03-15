function resolveStdlibMathFn(name, packageName) {
    if (typeof module !== 'undefined' && module.exports) {
        return require(packageName);
    }
    if (typeof globalThis !== 'undefined' && typeof globalThis[name] === 'function') {
        return globalThis[name];
    }
    throw new Error('Missing stdlib function: ' + name);
}

const max = resolveStdlibMathFn('max', '@stdlib/math-base-special-max');
const exp = resolveStdlibMathFn('exp', '@stdlib/math-base-special-exp');

function relu(z) {
    return max(0, z);
}

function reluPrime(z) {
    return z > 0 ? 1 : 0;
}

function softmaxInPlace(output) {
    let maxVal = -Infinity;
    for (let i = 0; i < output.length; i++) {
        maxVal = max(maxVal, output[i]);
    }

    let sum = 0;
    for (let i = 0; i < output.length; i++) {
        output[i] = exp(output[i] - maxVal);
        sum += output[i];
    }

    for (let i = 0; i < output.length; i++) {
        output[i] /= sum;
    }
    return output;
}

function softmaxFromLogits(values) {
    const out = Float64Array.from(values);
    softmaxInPlace(out);
    return out;
}

const api = {
    relu,
    reluPrime,
    softmaxInPlace,
    softmaxFromLogits
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
}

if (typeof globalThis !== 'undefined') {
    globalThis.StdlibMnistMath = api;
}