const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const ones = numeri.fill([3000, 3000], 1);

time(() => ones.sum({axes: [0]}), 'sum_axis', 88);
