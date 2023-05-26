// Fisher–Yates shuffle
export function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const temp = arr[j]
        arr[j] = arr[i]
        arr[i] = temp
    }
}
