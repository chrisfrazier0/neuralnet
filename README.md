# neuralnet

See example directory for XOR and MNIST examples.

```js
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
```

### Example Output

```
$ node examples/mnist.mjs
epoch  trainingCost  testCost  accuracy
1  0.588971805597505  0.11104161655039267  44.479191724801545
2  0.35613753193557995  0.11859514072648077  40.70242963675727
3  0.31868214037833453  0.0844840028866944  57.75799855665144
 ...
32  0.12615911685087475  0.01934087082030325  90.32956458984845
33  0.12522053593531743  0.018715419773875527  90.6422901130623
34  0.12016637310448795  0.017464517681020083  91.26774115949001
35  0.1288248763582059  0.017079624729372255  91.46018763531393
36  0.1268032120154334  0.017464517681020083  91.26774115949001
37  0.12015077169336179  0.020543661294202712  89.72816935289872
38  0.12761372152138006  0.017223959586240192  91.38802020687996
39  0.12655215150710217  0.020447438056290755  89.7762809718547
40  0.12526850658654115  0.017079624729372255  91.46018763531393

Network Accuracy:  91.46 %
```

![Training and test loss](/loss.png)
