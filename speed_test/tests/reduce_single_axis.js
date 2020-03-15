const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const ones = numeri.empty([3000, 2999]).map(() => Math.random());

time(() => ones.argmin({axis: 0}), 'argmin', 230);
