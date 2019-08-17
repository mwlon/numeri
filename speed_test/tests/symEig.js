const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const n = 100;
const randMat = numeri.empty([n, n]).mapInPlace(() => Math.random());
const symMat = randMat.plus(randMat.transpose());

time(() => numeri.symHessenberg(symMat, {includeQ: false}), 'warmup');
time(() => numeri.symHessenberg(symMat, {includeQ: false}), 'symHessenberg without q');
time(() => numeri.symHessenberg(symMat, {includeQ: true}), 'symHessenberg with q');
time(() => numeri.symEig(symMat, {includeVecs: false}), 'symEig without vecs');
time(() => numeri.symEig(symMat, {includeVecs: true}), 'symEig with vecs');
