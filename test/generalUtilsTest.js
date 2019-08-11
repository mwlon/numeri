const generalUtils = require('../lib/generalUtils');
const assert = require('assert');

describe('generalUtils', () => {
  describe('#prod', () => {
    it('should multiply an array of numbers', () => {
      assert.strictEqual(generalUtils.prod([1, 2, 3]), 6);
      assert.strictEqual(generalUtils.prod([1, 2, 3, 3]), 18);
      assert.strictEqual(generalUtils.prod([1, undefined, undefined, 3]), 3);
    });
  });

  describe('#defaults', () => {
    it('should overwrite defaults without modifying anything', () => {
      const options = {
        a: 1,
        c: 0
      };
      const defaults = {
        b: 2,
        c: 1
      };

      assert.deepEqual(generalUtils.defaults(options, defaults), {
        a: 1,
        b: 2,
        c: 0
      });

      assert.deepEqual(options, {a: 1, c: 0});
      assert.deepEqual(defaults, {b: 2, c: 1});
    });

    it('should accept undefined options', () => {
      const defaults = {
        b: 2,
        c: 1
      };

      assert.deepEqual(generalUtils.defaults(undefined, defaults), {
        b: 2,
        c: 1
      });
    });
  });
});
