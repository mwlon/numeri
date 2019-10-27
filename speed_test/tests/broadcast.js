const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const full = numeri.fill([3000, 3000], 1);

time(() => full.plus(1), 'broadcast_add', 72);
