const generalUtils = require('../lib/generalUtils');
const assert = require('assert');

describe('generalUtils', () => {
  describe('#prod', () => {
    it('should multiply an array of numbers', () => {
      assert.strictEqual(generalUtils.prod([1, 2, 3]), 6);
      assert.strictEqual(generalUtils.prod([1, 2, 3, 3]), 18);
    });
  });
});
