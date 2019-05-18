const assert = require('assert');
const { assertTensorEqual } = require('./testUtils');
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

  describe('#slice', () => {
    it('should support subsets and undefined (keep all)', () => {
      assertTensorEqual(
        createUtils.zeros([1, 3]).plus(notSimpleMat.slice([0, 1])),
        createUtils.fromFlat([1, 2, 3], [1, 3])
      );
    });

    it('should support choosing specific indices', () => {
      assertTensorEqual(
        createUtils.zeros([2]).plus(notSimpleMat.slice(undefined, 1)),
        createUtils.vector([2, 5])
      );
    });

    it('should support steps', () => {
      assertTensorEqual(
        createUtils.zeros([2, 2]).plus(notSimpleMat.slice(undefined, [0, 3, 2])),
        createUtils.fromFlat([1, 3, 4, 6], [2, 2])
      );
    });
  });

  describe('#transpose', () => {
    it('should transpose complicated things', () => {
      assertTensorEqual(
        createUtils.zeros([3, 2]).plus(notSimpleMat.transpose([1, 0])),
        createUtils.fromFlat([1, 4, 2, 5, 3, 6], [3, 2])
      );
    });

    it('should transpose things with >2 dimensions', () => {
      const mat3d = createUtils.fromFlat([0, 1, 2, 3, 4, 5], [1, 2, 3]);

      assertTensorEqual(
        createUtils.zeros([3, 1, 2]).plus(mat3d.transpose([1, 2, 0])),
        createUtils.fromFlat([0, 3, 1, 4, 2, 5], [3, 1, 2])
      );
    });
  });

  describe('#copy', () => {
    it('should return a new version of the data', () => {
      const copied = notSimpleMat.copy();

      assertTensorEqual(
        copied,
        simpleMat
      );

      copied.set([0, 0], 33);
      assert.strictEqual(copied.get(0, 0), 33);
      assert.strictEqual(notSimpleMat.get(0, 0), 1);
    });
  });

  describe('#plus', () => {
    it('should add numbers', () => {
      const expected = createUtils.fromFlat(
        [2, 4, 6, 8, 10, 12],
        [2, 3]
      );

      assertTensorEqual(
        simpleMat.plus(simpleMat),
        expected
      );

      assertTensorEqual(
        simpleMat.plus(notSimpleMat),
        expected
      );

      assertTensorEqual(
        notSimpleMat.plus(simpleMat),
        expected
      );
    });

    it('should work for scalars', () => {
      assertTensorEqual(
        createUtils.scalar(3).plus(createUtils.scalar(4)),
        createUtils.scalar(7)
      );
    });
  });

  describe('#minus', () => {
    it('should subtract numbers', () => {
      assertTensorEqual(
        createUtils.vector([1, 2]).minus(createUtils.vector([3, 4])),
        createUtils.vector([-2, -2])
      );
    });
  });

  describe('#times', () => {
    it('should multiply numbers', () => {
      assertTensorEqual(
        createUtils.vector([1, 2]).times(createUtils.vector([3, 4])),
        createUtils.vector([3, 8])
      );
    });
  });

  describe('#div', () => {
    it('should divide numbers', () => {
      assertTensorEqual(
        createUtils.vector([2, 6]).div(createUtils.vector([2, 3])),
        createUtils.vector([1, 2])
      );
    });
  });

  describe('#matMul', () => {
    it('should multiply matrices', () => {
      const colVec = createUtils.fromFlat(
        [0, 1, 2],
        [3, 1]
      );

      const expected = createUtils.fromFlat(
        [8, 17],
        [2, 1]
      );

      assertTensorEqual(
        notSimpleMat.matMul(colVec),
        expected
      );
    });
  });
});
