import { randn } from './random.mjs'

export class Matrix {
    constructor(rows = 0, cols = 0, init = 'zero') {
        if (Array.isArray(rows)) {
            init = cols
            cols = rows[1]
            rows = rows[0]
        }

        this.cache = null
        this.arr = Array(rows)
        for (let i = 0; i < rows; i++) {
            this.arr[i] = Array(cols)
            if (init === 'rand') {
                this.arr[i] = this.arr[i].fill().map(() => Math.random())
            } else if (init === 'randn' || init === 'normal') {
                this.arr[i] = this.arr[i].fill().map(() => randn())
            } else if (init === 'zero') {
                this.arr[i].fill(0)
            } else {
                this.arr[i].fill(init)
            }
        }
    }

    get rows() {
        return this.arr.length
    }

    get cols() {
        return this.arr[0].length
    }

    get size() {
        return this.rows * this.cols
    }

    get shape() {
        return [this.rows, this.cols]
    }

    get val() {
        return this.arr[0][0]
    }

    get T() {
        if (this.cache) {
            return this.cache
        }

        const result = new Matrix(this.cols, this.rows)
        for (let i = 0; i < this.cols; i++) {
            for (let j = 0; j < this.rows; j++) {
                result.arr[i][j] = this.arr[j][i]
            }
        }

        if (this.cache !== false) {
            this.cache = result
        }
        return result
    }

    map(cb) {
        const result = new Matrix(this.shape)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.arr[i][j] = cb(this.arr[i][j])
            }
        }
        return result
    }

    sum() {
        let result = 0
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result += this.arr[i][j]
            }
        }
        return result
    }

    eq(m) {
        if (!(m instanceof Matrix)) {
            m = new Matrix(this.shape, m)
        } else if (m.rows !== this.rows || m.cols !== this.cols) {
            return false
        }
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                if (this.arr[i][j] !== m.arr[i][j]) {
                    return false
                }
            }
        }
        return true
    }

    add(m) {
        if (!(m instanceof Matrix)) {
            m = new Matrix(this.shape, m)
        } else if (m.rows !== this.rows || m.cols !== this.cols) {
            throw Error('shape mismatch')
        }
        const result = new Matrix(this.shape)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.arr[i][j] = this.arr[i][j] + m.arr[i][j]
            }
        }
        return result
    }

    sub(m) {
        if (!(m instanceof Matrix)) {
            m = new Matrix(this.shape, m)
        } else if (m.rows !== this.rows || m.cols !== this.cols) {
            throw Error('shape mismatch')
        }
        const result = new Matrix(this.shape)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.arr[i][j] = this.arr[i][j] - m.arr[i][j]
            }
        }
        return result
    }

    // Hadamard product
    mul(m) {
        if (!(m instanceof Matrix)) {
            m = new Matrix(this.shape, m)
        } else if (m.rows !== this.rows || m.cols !== this.cols) {
            throw Error('shape mismatch')
        }
        const result = new Matrix(this.shape)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.arr[i][j] = this.arr[i][j] * m.arr[i][j]
            }
        }
        return result
    }

    // Hadamard division
    div(m) {
        if (!(m instanceof Matrix)) {
            m = new Matrix(this.shape, m)
        } else if (m.rows !== this.rows || m.cols !== this.cols) {
            throw Error('shape mismatch')
        }
        const result = new Matrix(this.shape)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.arr[i][j] = this.arr[i][j] / m.arr[i][j]
            }
        }
        return result
    }

    // Matrix multiplication
    dot(m) {
        if (!(m instanceof Matrix)) {
            return this.mul(m)
        } else if (this.cols !== m.rows) {
            throw Error('shape mismatch')
        }
        const result = new Matrix(this.rows, m.cols)
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                for (let k = 0; k < this.cols; k++) {
                    result.arr[i][j] += this.arr[i][k] * m.arr[k][j]
                }
            }
        }
        return result
    }
}

Matrix.from = function(arr) {
    if (!Array.isArray(arr)) {
        return new Matrix(1, 1, arr)
    }

    if (Array.isArray(arr[0])) {
        const m = new Matrix()
        m.arr = arr
        return m
    }

    const m = new Matrix(arr.length, 1)
    for (let i = 0; i < arr.length; i++) {
        m.arr[i][0] = arr[i]
    }
    return m
}
