const Tensor = require('./Tensor');
const createUtils = require('./createUtils');
const eig = require('./eig');

module.exports = Object.assign(
  {
    Tensor,
  },
  createUtils,
  eig
);
