const numeri = require('../../lib/index.js');
const { time, timeAndReturn } = require('../speedTestUtils');

const randMat = numeri.empty([4, 4]).mapInPlace(() => Math.random());
const symMat = numeri.fromNested(
[
  [
    1.2831304333375368,
    1.5064948504178346,
    0.6975059363393725
  ],
  [
    1.5064948504178346,
    1.5126195056218341,
    1.5764962094560167
  ],
  [
    0.6975059363393725,
    1.5764962094560167,
    1.9863777754573038
  ]
]
);//randMat.plus(randMat.transpose());
// console.log(symMat.toNested())

// time(() => numeri.symHessenberg(symMat, {includeQ: true}), 'symHessenbergWithQ');
// time(() => numeri.symHessenberg(symMat), 'symHessenberg');

// const {hessenberg, q} = numeri.symHessenberg(symMat, {includeQ: true});
// const hessenberg = numeri.fromNested([
//   [ 0.03170352398286491, -1.55458412889692, 0, 0 ],
//   [ -1.55458412889692, 2.5470006197826667, -1.4115800474044011, 0 ],
//   [ 0, -1.4115800474044011, 0.6737644350785483, -0.32609608341583973 ],
//   [ 0, 0, -0.32609608341583896, 0.22207347649366208 ]
// ])
// console.log(hessenberg.toNested());

const x = timeAndReturn(() => numeri.symEig(symMat, {includeVecs: true}), 'symEig');
console.log(x.vals.toNested());
console.log(x.vecs.toNested());
