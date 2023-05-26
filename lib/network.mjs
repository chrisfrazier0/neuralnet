import { Matrix } from './matrix.mjs'
import { shuffle } from './array.mjs'

export class Network {
    constructor(layers) {
        this.layers = layers
    }

    mse(actual, guess) {
        let cost = 0
        for (let i = 0; i < actual.rows; i++) {
            for (let j = 0; j < actual.cols; j++) {
                cost += ((actual.arr[i][j] - guess.arr[i][j])**2) / 2
            }
        }
        return cost / actual.size * 100
    }

    msePrime(actual, guess) {
        return actual.sub(guess)
    }

    train({ trainingData, testData, epochs, batchSize, learningRate, callback }) {
        epochs ||= 1
        batchSize ||= 32
        learningRate ||= 0.1

        for (let e = 0; e < epochs; e++) {
            let count = 0, cost = 0
            shuffle(trainingData)
            for (let i = 0; i < trainingData.length; i++) {
                const input = trainingData[i].input instanceof Matrix
                    ? trainingData[i].input : Matrix.from(trainingData[i].input)
                const solution = trainingData[i].solution instanceof Matrix
                    ? trainingData[i].solution : Matrix.from(trainingData[i].solution)

                // forward
                let output = input
                for (const layer of this.layers) {
                    if (layer.util) continue
                    output = layer.forward(output)
                }

                // cost
                cost += this.mse(solution, output)

                // backward
                let gradient = this.msePrime(solution, output)
                for (const layer of this.layers.toReversed()) {
                    if (layer.util) continue
                    gradient = layer.backward(gradient, learningRate)
                }
                count += 1

                // mini-batch
                if (count === batchSize || i === trainingData.length - 1) {
                    for (const layer of this.layers) {
                        if (layer.apply) layer.apply()
                    }
                    count = 0
                }
            }

            if (callback) {
                let acc = 0
                if (testData) {
                    shuffle(testData)
                    acc = this.test(testData.slice(0, 111))
                }
                callback.call(null, e, epochs, cost / trainingData.length, acc)
            }
        }
    }

    test(testData) {
        let correct = 0
        for (let i = 0; i < testData.length; i++) {
            const input = testData[i].input instanceof Matrix
                ? testData[i].input : Matrix.from(testData[i].input)
            const solution = testData[i].solution instanceof Matrix
                ? testData[i].solution : Matrix.from(testData[i].solution)

            let output = input
            for (const layer of this.layers) {
                output = layer.forward(output)
            }

            if (output.eq(solution)) {
                correct += 1
            }
        }
        return correct / testData.length * 100
    }

    run(...args) {
        let output = Matrix.from(args)
        for (const layer of this.layers) {
            output = layer.forward(output)
        }
        return output.size !== 1 ? output : output.arr[0][0]
    }
}
