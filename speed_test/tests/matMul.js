const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const small = numeri.fill([200, 400], 1);
const another = numeri.fill([400, 200], 1);
time(() => small.matMul(another), 'mat_mul');
time(() => small.matMul(small.transpose()), 'mat_mul_transposed');
