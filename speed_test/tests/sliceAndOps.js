const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const ones = numeri.fill([3000, 4000], 1);

const sliced = timeAndReturn(() => ones.slice(undefined, [500, 3500]), 'slice');
const transposed = timeAndReturn(() => sliced.transpose(), 'transpose');
time(() => transposed.copy(), 'copy');
time(() => transposed.plus(transposed), 'complicated_add');
