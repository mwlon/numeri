const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const n = 100;
const randMat = numeri.empty([n, n]).mapInPlace(() => Math.random());
const symMat = randMat.plus(randMat.transpose());

time(() => numeri.symHessenberg(symMat, {includeQ: false}), 'warmup', 170);
time(() => numeri.symHessenberg(symMat, {includeQ: false}), 'symHessenberg without q', 92);
time(() => numeri.symHessenberg(symMat, {includeQ: true}), 'symHessenberg with q', 152);
time(() => numeri.symEig(symMat, {includeVecs: false}), 'symEig without vecs', 109);
time(() => numeri.symEig(symMat, {includeVecs: true}), 'symEig with vecs', 217);
