const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const rands = numeri.empty([3000, 2999]).map(() => Math.random());

// time(() => {
//   const data = ones.data;
//   const result = Array(3000);
//   for (let i = 0; i < 3000; i++) {
//     const base = i * 2999;
//     let min = Infinity;
//     let minJ = -1;
//     let s = 0;
//     for (let j = 0; j < 2999; j++) {
//       const dataJ = base + j;
//       const val = data[dataJ];
//       s += val;
//       // if (val < min) {
//       //   min = val;
//       //   minJ = j;
//       // }
//     }
//     // result[i] = minJ;
//     result[i] = s;
//   }
//   console.log(result);
// });
// time(() => {
//   const result = Array(3000);
//   const mins = Array(3000).fill(Infinity);
//   const data = ones.data;
//   for (let i = 0; i < 3000 * 2999; i++) {
//     const resultJ = Math.floor(i / 3000);
//     const val = data[i];
//     if (val < mins[resultJ]) {
//       const axisJ = i % 2999;
//       mins[resultJ] = val;
//       result[resultJ] = axisJ;
//     }
//   }
//   console.log(result);
// });
time(() => rands.argmin({axis: 1}), 'argmin', 230);
