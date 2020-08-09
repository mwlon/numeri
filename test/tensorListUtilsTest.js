const assert = require('assert');
const createUtils = require('../lib/createUtils');
const { stack, concat } = require('../lib/tensorListUtils');

describe('tensorListUtils', () => {
  describe('concat', () => {
    it('stacks em up real good', () => {
      const t0 = createUtils.fill([2, 3, undefined, undefined], 0);
      const t1 = createUtils.fill([undefined, 1, 4, undefined], 1);
      const t2 = createUtils.fill([2, 3, undefined, undefined], 2);
      const stacked = concat([t0, t1, t2], 1);
      assert.deepEqual(stacked.shape, [2, 7, 4, undefined]);
      assert.strictEqual(stacked.get(0, 0, 0, 0), 0);
      assert.strictEqual(stacked.get(1, 0, 1, 1), 0);
      assert.strictEqual(stacked.get(0, 2, 0, 0), 0);
      assert.strictEqual(stacked.get(0, 3, 0, 0), 1);
      assert.strictEqual(stacked.get(0, 4, 0, 0), 2);
      assert.strictEqual(stacked.get(0, 6, 0, 0), 2);
    });
  });

  describe('stack', () => {
    it('stacks em up real good', () => {
      const t0 = createUtils.fill([2, undefined, undefined], 0);
      const t1 = createUtils.fill([undefined, 4, undefined], 1);
      const t2 = createUtils.fill([2, undefined, undefined], 2);
      const stacked = stack([t0, t1, t2], 1);
      assert.deepEqual(stacked.shape, [2, 3, 4, undefined]);
      assert.strictEqual(stacked.get(0, 0, 0, 0), 0);
      assert.strictEqual(stacked.get(1, 0, 1, 1), 0);
      assert.strictEqual(stacked.get(0, 1, 0, 0), 1);
      assert.strictEqual(stacked.get(0, 2, 0, 0), 2);
    });
  });
});
