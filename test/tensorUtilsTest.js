const assert = require('assert');
const tensorUtils = require('../lib/tensorUtils');

describe('tensorUtils', () => {
  describe('#getStridesAndMods', () => {
    it('should give the right strides and mods', () => {
      assert.deepEqual(
        tensorUtils.getStridesAndMods([7, 5, 2]),
        {
          strides: [10, 2, 1],
          mods: [70, 10, 2]
        }
      );
    });
  });

  describe('#shapesEqual', () => {
    it('should return true if equal and false otherwise', () => {
      assert(!tensorUtils.shapesEqual([1, 2, 3], [1, 2]));
      assert(!tensorUtils.shapesEqual([1, 2], [1, 2, 3]));
      assert(!tensorUtils.shapesEqual([1, 2, 3], [1, 2, 4]));
      assert(!tensorUtils.shapesEqual([1, 2, 3], [1, 2, undefined]));
      assert(tensorUtils.shapesEqual([1, 2, 3], [1, 2, 3]));
    });
  });
});
