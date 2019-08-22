const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const ones = numeri.fill([3000, 4000], 1);

const sliced = timeAndReturn(() => ones.slice(undefined, [500, 3500]), 'slice', 1);
const transposed = timeAndReturn(() => sliced.transpose(), 'transpose', 1);
time(() => transposed.copy(), 'copy', 440);
time(() => transposed.plus(transposed), 'complicated_add', 698);
