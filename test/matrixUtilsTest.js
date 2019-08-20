const assert = require('assert');
const { assertTensorEqual } = require('./testUtils');
const Tensor = require('../lib/Tensor');
const createUtils = require('../lib/createUtils');
const { matMul, vecMulMat, matMulVec, dot, outerProd } = require('../lib/matrixUtils');

describe('matrixUtils', () => {
  const notSimpleMat = new Tensor({
    data: [-1000, 1, 4, 2, 5, 3, 6, -3000],
    shape: [2, 3],
    strides: [1, 2],
    mods: [2, 6],
    offset: 1
  });
  //1 2 3
  //4 5 6

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
        matMul(notSimpleMat, colVec),
        expected
      );
    });

    it('should infer vectors as row/col', () => {
      const colVec = createUtils.vector([0, 1, 2]);
      assertTensorEqual(
        matMul(notSimpleMat, colVec),
        createUtils.vector([8, 17])
      );
      assertTensorEqual(
        matMulVec(notSimpleMat, colVec),
        createUtils.vector([8, 17])
      );

      const rowVec = createUtils.vector([3, 4]);
      assertTensorEqual(
        matMul(rowVec, notSimpleMat),
        createUtils.vector([19, 26, 33])
      );
      assertTensorEqual(
        vecMulMat(rowVec, notSimpleMat),
        createUtils.vector([19, 26, 33])
      );
    });
  });

  describe('#outerProd', () => {
    it('should do combinations of multiplication', () => {
      assertTensorEqual(
        outerProd(createUtils.vector([0, 1, 2]), createUtils.vector([3, 4])),
        createUtils.fromFlat([0, 0, 3, 4, 6, 8], [3, 2])
      );
    });
  });

  describe('#dot', () => {
    it('should take dot product of vectors', () => {
      const vec1 = createUtils.vector([1, 2, 3]);
      const vec2 = createUtils.vector([2, 3, 4]);

      assert.strictEqual(dot(vec1, vec2), 20);
    });
  });
});
