const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const ones = numeri.fill([3000, 3000], 1);

time(() => ones.sum(), 'sum_to_number', 75);
// time(() => ones.sum({keepScalarAsTensor: true}), 'sum_to_scalar');
// time(() => ones.sum({axes: [0]}), 'sum_axis_0');
// time(() => ones.sum({axes: [1]}), 'sum_axis_1');
