const Tensor = require('./Tensor');
const createUtils = require('./createUtils');

module.exports = {
  Tensor,
  scalar: createUtils.scalar,
  vector: createUtils.vector,
  fromFlat: createUtils.fromFlat,
  fromNested: createUtils.fromNested,
  zeros: createUtils.zeros,
  fill: createUtils.fill,
};
