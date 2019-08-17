const Tensor = require('./Tensor');
const createUtils = require('./createUtils');
const eig = require('./eig');
const matrixUtils = require('./matrixUtils');

module.exports = Object.assign(
  {
    Tensor,
  },
  createUtils,
  eig,
  matrixUtils
);
