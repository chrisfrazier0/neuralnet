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
    brain.layer.output({ size: 1, activation: 'tanh', util: 'round' }),
)

network.train({
    trainingData: data,
    epochs: 900,
    learningRate: 0.4,

    callback(e, total, cost) {
        console.log('%d/%d\tcost = %f', e+1, total, cost)
    },
})

console.log('\nXOR(0,0) = ', network.run(0, 0))
console.log('XOR(0,1) = ',   network.run(0, 1))
console.log('XOR(1,0) = ',   network.run(1, 0))
console.log('XOR(1,1) = ',   network.run(1, 1))
