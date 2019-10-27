const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const small = numeri.fill([300, 300], 1);
time(() => numeri.matMul(small, small), 'mat_mul', 100);
