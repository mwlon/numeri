const assert = require('assert');

module.exports = {
  assertTensorEqual(tens0, tens1, tolerance=0) {
    assert.deepEqual(tens0.shape, tens1.shape, 'shape');
    assert.deepEqual(tens0.offset, tens1.offset, 'offset');
    assert.deepEqual(tens0.strides, tens1.strides, 'strides');
    assert.deepEqual(tens0.mods, tens1.mods, 'mods');
    if (!module.exports.dataWithinTolerance(tens0.data, tens1.data, tolerance)) {
      throw new assert.AssertionError({
        actual: tens0.data,
        expected: tens1.data,
        message: `tensor data did not match within tolerance of ${tolerance}`
      });
    }
  },

  dataWithinTolerance(data0, data1, tolerance) {
    for (var j = 0; j < data0.length; j++) {
      const diff = Math.abs(data0[j] - data1[j]);
      if (diff > tolerance || isNaN(diff) || diff === undefined) {
        return false;
      }
    }
    return true;
  }
};
