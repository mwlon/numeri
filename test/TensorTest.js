const assert = require('assert');
const Tensor = require('../lib/Tensor');
const createUtils = require('../lib/createUtils');

describe('Tensor', () => {
  const simpleMat = createUtils.fromFlat(
    [1, 2, 3, 4, 5, 6],
    [2, 3]
  );
  const notSimpleMat = new Tensor({
    data: [-1000, 1, 4, 2, 5, 3, 6, -3000],
    shape: [2, 3],
    strides: [1, 2],
    mods: [2, 6],
    offset: 1
  });

  describe('#plus', () => {
    it('should add numbers', () => {
      const expected = createUtils.fromFlat(
        [2, 4, 6, 8, 10, 12],
        [2, 3]
      );

      assert.deepEqual(
        simpleMat.plus(simpleMat),
        expected
      );

      assert.deepEqual(
        simpleMat.plus(notSimpleMat),
        expected
      );

      assert.deepEqual(
        notSimpleMat.plus(simpleMat),
        expected
      );
    });
  });

  describe('#minus', () => {
    it('should subtract numbers', () => {
      assert.deepEqual(
        createUtils.vector([1, 2]).minus(createUtils.vector([3, 4])),
        createUtils.vector([-2, -2])
      );
    });
  });

  describe('#times', () => {
    it('should multiply numbers', () => {
      assert.deepEqual(
        createUtils.vector([1, 2]).times(createUtils.vector([3, 4])),
        createUtils.vector([3, 8])
      );
    });
  });

  describe('#div', () => {
    it('should divide numbers', () => {
      assert.deepEqual(
        createUtils.vector([2, 6]).div(createUtils.vector([2, 3])),
        createUtils.vector([1, 2])
      );
    });
  });

  describe('#slice', () => {
    it('should support subsets and undefined (keep all)', () => {
      assert.deepEqual(
        createUtils.zeros([1, 3]).plus(notSimpleMat.slice([0, 1])),
        createUtils.fromFlat([1, 2, 3], [1, 3])
      );
    });

    it('should support choosing specific indices', () => {
      assert.deepEqual(
        createUtils.zeros([2]).plus(notSimpleMat.slice(undefined, 1)),
        createUtils.vector([2, 5])
      );
    });

    it('should support steps', () => {
      assert.deepEqual(
        createUtils.zeros([2, 2]).plus(notSimpleMat.slice(undefined, [0, 3, 2])),
        createUtils.fromFlat([1, 3, 4, 6], [2, 2])
      );
    });
  });

  describe('#transpose', () => {
    it('should transpose complicated things', () => {
      assert.deepEqual(
        createUtils.zeros([3, 2]).plus(notSimpleMat.transpose([1, 0])),
        createUtils.fromFlat([1, 4, 2, 5, 3, 6], [3, 2])
      );
    });

    it('should transpose things with >2 dimensions', () => {
      const mat3d = createUtils.fromFlat([0, 1, 2, 3, 4, 5], [1, 2, 3]);

      assert.deepEqual(
        createUtils.zeros([3, 1, 2]).plus(mat3d.transpose([1, 2, 0])),
        createUtils.fromFlat([0, 3, 1, 4, 2, 5], [3, 1, 2])
      );
    });
  });
});
