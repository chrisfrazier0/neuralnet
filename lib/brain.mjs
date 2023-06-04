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

        dense({ size, activation, useBias }) {
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
}

// aliases
brain.layer.output = brain.layer.dense
