const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const ones = numeri.empty([3000, 2999]).map(() => Math.random());

const x = timeAndReturn(() => ones.argmin({axis: 0}), 'argmin', 230);
