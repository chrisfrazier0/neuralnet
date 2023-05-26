# neuralnet

See example directory for XOR and MNIST examples.

```js
const trainingData = mnistLoad({ filename: './data/mnist_train.csv', digits, limit: 500 })
const testData = mnistLoad({ filename: './data/mnist_test.csv', digits })

const network = brain.network(
    brain.layer.input({ size: 784 }),
    brain.layer.dense({ size: 16, activation: 'tanh' }),
    brain.layer.dense({ size: 16, activation: 'tanh' }),
    brain.layer.output({ size: 10, activation: 'tanh', util: 'onehot' }),
)

network.train({
    trainingData,
    testData,
    epochs: 40,
    learningRate: 0.18,
})

console.log('\nNetwork Accuracy: ', network.test(testData).toFixed(2), '%')
```
