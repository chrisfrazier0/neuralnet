<a name="readme-top"></a>

# neuralnet

<!-- BADGES -->

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- ABOUT THE PROJECT -->

## About The Project

This is an object oriented feed forward neural network implementation completed during
a codejam over the course of one weekend. The implementation focuses on ease of use and
understanding over performance.

[![Javascript][javascript]][javascript-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Repository includes full examples for solving XOR and training on the MNIST data set.

### MNIST Example

```js
const trainingData = mnistLoad({
  filename: "./data/mnist_train.csv",
  digits,
  limit: 500,
});
const testData = mnistLoad({ filename: "./data/mnist_test.csv", digits });

const network = brain.network(
  brain.layer.input({ size: 784 }),
  brain.layer.dense({ size: 16, activation: "tanh" }),
  brain.layer.dense({ size: 16, activation: "tanh" }),
  brain.layer.output({ size: 10, activation: "tanh" })
);

console.log("epoch  trainingCost  testCost  accuracy");

network.train({
  trainingData,
  testData,
  method: "onehot",
  epochs: 40,
  learningRate: 0.18,

  callback(e, trainingCost, testCost, acc) {
    console.log("%d  %d  %d  %d", e + 1, trainingCost, testCost, acc);
  },
});

console.log(
  "\nNetwork Accuracy: ",
  network.test(testData, "onehot")[1].toFixed(2),
  "%"
);
```

#### Output

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

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/chrisfrazier0/neuralnet.svg?style=for-the-badge
[license-url]: https://github.com/chrisfrazier0/neuralnet/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/chrisfrazier0
[javascript]: https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black
[javascript-url]: https://developer.mozilla.org/en-US/docs/Web/JavaScript
