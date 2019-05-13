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
});
