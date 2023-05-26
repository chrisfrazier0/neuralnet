// Box–Muller transform
export function randn({ mean = 0, stdev = 1 } = {}) {
    const u = Math.random()
    const v = 1 - Math.random()
    const z = Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v)
    return z * stdev + mean
}
