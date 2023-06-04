import { brain } from '../lib/brain.mjs'

const data = [
    { input: [0, 0], solution: 0 },
    { input: [0, 1], solution: 1 },
    { input: [1, 0], solution: 1 },
    { input: [1, 1], solution: 0 },
]

const network = brain.network(
    brain.layer.input({ size: 2 }),
    brain.layer.dense({ size: 3, activation: 'tanh' }),
    brain.layer.output({ size: 1, activation: 'tanh' }),
)

const epochs = 900

network.train({
    trainingData: data,
    epochs,
    learningRate: 0.4,

    callback(e, trainingCost) {
        console.log('%d/%d\tcost = %f', e+1, epochs, trainingCost)
    },
})

console.log('\nXOR(0,0) = ', network.round(0, 0).val)
console.log('XOR(0,1) = ',   network.round(0, 1).val)
console.log('XOR(1,0) = ',   network.round(1, 0).val)
console.log('XOR(1,1) = ',   network.round(1, 1).val)
