const assert = require('assert');

module.exports = {
  assertTensorEqual(tens0, tens1) {
    assert.deepEqual(tens0.data, tens1.data, 'data');
    assert.deepEqual(tens0.shape, tens1.shape, 'shape');
    assert.deepEqual(tens0.offset, tens1.offset, 'offset');
    assert.deepEqual(tens0.strides, tens1.strides, 'strides');
    assert.deepEqual(tens0.mods, tens1.mods, 'mods');
  }
};
