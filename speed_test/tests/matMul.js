const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const small = numeri.fill([200, 400], 1);
time(() => numeri.matMul(small, small.transpose()), 'mat_mul_transposed', 49);
