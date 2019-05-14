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

  checkSameShape(shape0, shape1) {
    assert.deepEqual(
      shape0,
      shape1,
      `Tensors have different shapes: ${shape0} vs ${shape1}`
    );
  },

  checkMatMulShape(shape0, shape1) {
    for (var shape of [shape0, shape1]) {
      assert.strictEqual(
        shape.length,
        2,
        `Cannot perform matrix multiplication on ${shape.length}-d tensor of shape ${shape}`
      );
    }

    assert.strictEqual(
      shape0[1],
      shape1[0],
      `Cannot perform matrix multiplication mismatched shapes ${shape0}, ${shape1}`
    );
  }
};