const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const ones = numeri.fill([3000 * 4000], 1);

const sliced = ones.slice([3000 * 3000, 0, -1]);
time(() => sliced.copy(), 'copy_vector', 80);
