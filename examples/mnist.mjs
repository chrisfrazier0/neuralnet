import { mnistLoad } from '../data/mnistLoad.mjs'
import { brain } from '../lib/brain.mjs'

const digits = [0, 1, 2, 3]
const trainingData = mnistLoad({ filename: './data/mnist_train.csv', digits, limit: 500 })
const testData = mnistLoad({ filename: './data/mnist_test.csv', digits })

const network = brain.network(
    brain.layer.input({ size: 784 }),
    brain.layer.dense({ size: 16, activation: 'tanh' }),
    brain.layer.dense({ size: 16, activation: 'tanh' }),
    brain.layer.output({ size: 10, activation: 'tanh' }),
)

console.log('epoch  trainingCost  testCost  accuracy')

network.train({
    trainingData,
    testData,
    method: 'onehot',
    epochs: 40,
    learningRate: 0.18,

    callback(e, trainingCost, testCost, acc) {
        console.log('%d  %d  %d  %d', e+1, trainingCost, testCost, acc)
    },
})

console.log('\nNetwork Accuracy: ', network.test(testData, 'onehot')[1].toFixed(2), '%')
