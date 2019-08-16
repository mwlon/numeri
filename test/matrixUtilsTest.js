const { assertTensorEqual } = require('./testUtils');
const Tensor = require('../lib/Tensor');
const createUtils = require('../lib/createUtils');
const { matMul } = require('../lib/matrixUtils');

describe('matrixUtils', () => {
  const notSimpleMat = new Tensor({
    data: [-1000, 1, 4, 2, 5, 3, 6, -3000],
    shape: [2, 3],
    strides: [1, 2],
    mods: [2, 6],
    offset: 1
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
        matMul(notSimpleMat, colVec),
        expected
      );
    });
  });
});
