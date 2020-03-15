const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const highD = numeri.fill([5, 5, 5, 5, 5], 1);

time(() => {
  let s = 0;
  for (let iter = 0; iter < 10000000; iter++) {
    s += highD.get(iter % 2, 0, 1, 2, 3);
  }
  return s;
}, 'get_high_dimensions', 25);

const matrix = numeri.fill([5, 5], 2);
time(() => {
  let s = 0;
  for (let iter = 0; iter < 10000000; iter++) {
    s += matrix.get(1, 0);
  }
  return s;
}, 'get_matrix', 14);
