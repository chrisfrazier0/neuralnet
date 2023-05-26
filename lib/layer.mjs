import { Matrix } from './matrix.mjs'

export class Dense {
    constructor(weights, bias) {
        this.count = 0
        this.input = null
        this.weights = weights
        this.weightsGradient = new Matrix(weights.shape)
        this.bias = bias
        this.biasGradient = bias && new Matrix(bias.shape)
    }

    forward(input) {
        this.input = input
        return this.weights.dot(input).add(this.bias || 0)
    }

    backward(outputGradient, learningRate) {
        this.weightsGradient = this.weightsGradient.add(outputGradient.dot(this.input.T).mul(learningRate))
        if (this.bias) {
            this.biasGradient = this.biasGradient.add(outputGradient.mul(learningRate))
        }
        this.count += 1
        return this.weights.T.dot(outputGradient)
    }

    apply() {
        this.weights = this.weights.add(this.weightsGradient.div(this.count))
        this.bias = this.bias.add(this.biasGradient.div(this.count))

        this.count = 0
        this.weightsGradient = new Matrix(this.weights.shape)
        this.biasGradient = new Matrix(this.bias.shape)
    }
}

class Activation {
    constructor(func, funcPrime) {
        this.input = null
        this.func = func
        this.funcPrime = funcPrime
    }

    forward(input) {
        this.input = input
        return input.map(this.func)
    }

    backward(outputGradient) {
        return this.input.map(this.funcPrime).mul(outputGradient)
    }
}

export class Sigmoid extends Activation {
    constructor() {
        const sigmoid = x => 1 / (1 - Math.exp(-x))
        const sigmoidPrime = x => {
            const s = sigmoid(x)
            return s * (1 - s)
        }
        super(sigmoid, sigmoidPrime)
    }
}

export class Tanh extends Activation {
    constructor() {
        const tanh = x => Math.tanh(x)
        const tanhPrime = x => 1 - Math.tanh(x)**2
        super(tanh, tanhPrime)
    }
}

export class ReLU extends Activation {
    constructor() {
        const relu = x => Math.max(0, x)
        const reluPrime = x => x > 0 ? 1 : 0
        super(relu, reluPrime)
    }
}
