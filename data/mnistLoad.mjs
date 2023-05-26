import fs from 'node:fs'
import { Matrix } from '../lib/matrix.mjs'
import { shuffle } from '../lib/array.mjs'

function onehot(n) {
    const result = new Matrix(10, 1)
    result.arr[n][0] = 1
    return result
}

function processLine(line, normalize) {
    const arr = line.split(',').map(x => parseInt(x))
    return {
        label: arr[0],
        input: normalize
            ? Matrix.from(arr.slice(1).map(x => x / 255))
            : Matrix.from(arr.slice(1)),
        solution: onehot(arr[0]),
    }
}

export function mnistLoad({ filename, digits, limit, normalize }) {
    digits ||= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if (normalize === undefined) normalize = true

    const data = fs.readFileSync(filename, 'utf8')
        .split(/\r?\n/)
        .slice(0, -1)
        .map(line => processLine(line, normalize))
    shuffle(data)

    return digits.map(digit =>
        data.filter(x => x.label === digit).slice(0, limit)
    ).flat()
}
