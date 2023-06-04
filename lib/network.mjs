import { Matrix } from './matrix.mjs'
import { shuffle } from './array.mjs'

export class Network {
    constructor(layers) {
        this.layers = layers
    }

    mse(prediction, actual) {
        let cost = 0
        for (let i = 0; i < prediction.rows; i++) {
            for (let j = 0; j < prediction.cols; j++) {
                cost += (prediction.arr[i][j] - actual.arr[i][j])**2
            }
        }
        return cost / prediction.size
    }

    msePrime(prediction, actual) {
        return prediction.sub(actual).mul(2)
    }

    train({ trainingData, testData, method, epochs, batchSize, learningRate, callback }) {
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
                    output = layer.forward(output)
                }

                // cost
                cost += this.mse(output, solution)

                // backward
                let gradient = this.msePrime(output, solution)
                for (const layer of this.layers.toReversed()) {
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
                let testCost = 0, acc = 0
                if (testData) {
                    ;[testCost, acc] = this.test(testData, method)
                }
                callback.call(null, e, cost / trainingData.length, testCost, acc)
            }
        }
    }

    test(testData, method = 'run') {
        let correct = 0, cost = 0
        for (let i = 0; i < testData.length; i++) {
            const input = testData[i].input instanceof Matrix
                ? testData[i].input : Matrix.from(testData[i].input)
            const solution = testData[i].solution instanceof Matrix
                ? testData[i].solution : Matrix.from(testData[i].solution)

            let output = null
            if (method === 'round') {
                output = this.round(...input.T.arr[0])
            } else if (method === 'onehot') {
                output = this.onehot(...input.T.arr[0])
            } else {
                output = this.run(...input.T.arr[0])
            }

            cost += this.mse(output, solution)

            if (output.eq(solution)) {
                correct += 1
            }
        }
        return [cost / testData.length, correct / testData.length * 100]
    }

    run(...args) {
        let output = Matrix.from(args)
        for (const layer of this.layers) {
            output = layer.forward(output)
        }
        return output
    }

    round(...args) {
        const output = this.run(...args)
        return output instanceof Matrix
            ? output.map(x => Math.abs(Math.round(x)))
            : Math.abs(Math.round(output))
    }

    onehot(...args) {
        const output = this.run(...args)
        let max = output.arr[0][0]
        let maxIndex = 0
        for (let i = 0; i < output.rows; i++) {
            if (output.arr[i][0] > max) {
                max = output.arr[i][0]
                maxIndex = i
            }
        }
        const result = new Matrix(output.shape)
        result.arr[maxIndex][0] = 1
        return result
    }
}
