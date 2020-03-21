const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const highD = numeri.fill([5, 5, 5, 5, 5], 1);

time(() => {
  for (let iter = 0; iter < 1000000; iter++) {
    highD.update([0, 1, 2, 3, 4], (x) => x + 1);
  }
}, 'update_high_dimensions', 34);
