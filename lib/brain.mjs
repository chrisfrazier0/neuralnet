import { Matrix } from '../lib/matrix.mjs'
import { Network } from './network.mjs'
import { Dense, ReLU, Sigmoid, Tanh } from '../lib/layer.mjs'

// This is a collection of factory methods designed to expose a simple API
export const brain = {
    network(...factories) {
        let layers = [], inputSize = factories.shift()
        for (const factory of factories) {
            layers = layers.concat(factory(inputSize))
            inputSize = factory.outputSize || inputSize
        }
        return new Network(layers)
    },

    layer: {
        input({ size }) {
            return size
        },

        dense({ size, activation, useBias, util }) {
            const factory = function(inputSize) {
                const result = []
                const weights = new Matrix(size, inputSize, 'normal')
                const bias = useBias !== false ? new Matrix(size, 1, 'normal') : null
                result.push(new Dense(weights, bias))

                let actv = null
                if (activation === 'sigmoid') {
                    actv = new Sigmoid()
                } else if (activation === 'tanh') {
                    actv = new Tanh()
                } else if (activation === 'relu') {
                    actv = new ReLU()
                }

                if (actv) result.push(actv)
                if (util) result.push(brain.util[util]()())
                return result
            }
            factory.outputSize = size
            return factory
        },

        sigmoid() {
            return () => new Sigmoid()
        },

        tanh() {
            return () => new Tanh()
        },

        relu() {
            return () => new ReLU()
        },
    },

    util: {
        round() {
            return () => ({
                util: true,
                forward: input => input.map(x => {
                    const round = Math.round(x)
                    return 1 / round === -Infinity ? 0 : round
                }),
            })
        },

        onehot() {
            return () => ({
                util: true,
                forward: input => {
                    let max = input.arr[0][0]
                    let maxIndex = 0
                    for (let i = 0; i < input.rows; i++) {
                        if (input.arr[i][0] > max) {
                            max = input.arr[i][0]
                            maxIndex = i
                        }
                    }
                    const result = new Matrix(input.shape)
                    result.arr[maxIndex][0] = 1
                    return result
                },
            })
        },
    },
}

// aliases
brain.layer.output = brain.layer.dense
