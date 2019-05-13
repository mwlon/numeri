const assert = require('assert');
const generalUtils = require('./generalUtils');
const Tensor = require('./Tensor');

module.exports = {
  getStridesAndMods(shape) {
    const dims = shape.length;
    let stride = 1;
    const mods = Array(dims);
    const strides = Array(dims);

    for (var idx = dims - 1; idx >= 0; idx--) {
      strides[idx] = stride;
      stride *= shape[idx];
      mods[idx] = stride;
    }

    return {
      strides,
      mods
    };
  },
};