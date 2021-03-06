const assert = require('assert');
const { assertTensorEqual } = require('./testUtils');
const createUtils = require('../lib/createUtils');
const eig = require('../lib/eig');
const { matMul } = require('../lib/matrixUtils');

describe('eig', () => {
  const n = 4;
  const symMat = createUtils.fromFlat(
    [1, 2, 3, 4,
      2, 5, 6, 7,
      3, 6, 8, 9,
      4, 7, 9, 10],
    [n, n]
  );

  describe('#symHessenberg', () => {
    it('returns a tridiagonal matrix with or without getting Q matrix', () => {
      const { hessenberg, q } = eig.symHessenberg(symMat, {includeQ: true});
      assertTensorEqual(
        matMul(matMul(q, hessenberg), q.transpose()),
        symMat,
        1E-12
      );
      assertTensorEqual(
        matMul(q, q.transpose()),
        createUtils.identity(n),
        1E-12
      );
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (Math.abs(i - j) > 1) {
            //should be 0's off the tridiagonal
            assert.strictEqual(hessenberg.get(i, j), 0);
          } else {
            //on the tridiagonal
            assert(Math.abs(hessenberg.get(i, j)) > 1E-4);
          }
        }
      }

      const resultsWithoutQ = eig.symHessenberg(symMat, {includeQ: false});
      assert(!resultsWithoutQ.q);
    });

    it('works on matrices with 0s in unhelpful places', () => {
      const unhelpfulMat = createUtils.fromFlat([
        0, 0, 1,
        0, 2, 4,
        1, 4, 0
      ], [3, 3]);
      const { hessenberg, q } = eig.symHessenberg(unhelpfulMat, {includeQ: true});
      const expected = createUtils.fromFlat([
        0, -1, 0,
        -1, 0, -4,
        0, -4, 2
      ], [3, 3]);
      assertTensorEqual(
        hessenberg,
        expected,
        1E-12
      );
      assertTensorEqual(
        matMul(matMul(q, hessenberg), q.transpose()),
        unhelpfulMat,
        1E-12
      );
    });
  });

  describe('#getQrShift', () => {
    it('returns the min eigenvalue by absolute value', () => {
      assert.strictEqual(
        eig.getQrShift(1, 0, 0, 2),
        1,
        'same sign case'
      );
      assert.strictEqual(
        eig.getQrShift(-1, 0, 0, 2),
        -1,
        'diff sign case'
      );
      assert.strictEqual(
        eig.getQrShift(2, 0, 0, -1),
        -1,
        '2nd diff sign case'
      );
      assert.strictEqual(
        eig.getQrShift(2, 1, 1, 2),
        1,
        'bc terms case'
      );
      assert.strictEqual(
        eig.getQrShift(3, 1, 4, 3),
        1,
        'nonequal bc terms case'
      );
      assert.strictEqual(
        Math.abs(eig.getQrShift(0, -2, -2, 0)),
        2,
        'zero a + d case'
      );
    });
  });

  describe('#symEig', () => {
    it('returns the eigenvalues and vectors', () => {
      const mats = [
        symMat,
        createUtils.fromFlat([
          0, 0, 2,
          0, 3, 0,
          2, 0, 0
        ], [3, 3]),
        createUtils.fromFlat([
          1, 1, 1,
          1, 1, 0,
          1, 0, 1
        ], [3, 3]),
        createUtils.fromFlat([
          1, -2,
          -2, 1
        ], [2, 2]),
        createUtils.fromFlat([3], [1, 1])
      ];

      mats.forEach((mat) => {
        const { vals, vecs } = eig.symEig(mat, {includeVecs: true});
        assertTensorEqual(
          matMul(matMul(vecs, createUtils.diagonal(vals.data)), vecs.transpose()),
          mat,
          1E-6
        );
      });

      const resultsWithoutVecs = eig.symEig(symMat, {includeVecs: false});
      assert(!resultsWithoutVecs.vecs);
    });

    it('works on a random, large-ish matrix', () => {
      const randMat = createUtils.empty([10, 10]).mapInPlace(() => Math.random());
      const randSymMat = randMat.plus(randMat.transpose());
      const { vals, vecs } = eig.symEig(randSymMat, {includeVecs: true});
      assertTensorEqual(
        matMul(matMul(vecs, createUtils.diagonal(vals.data)), vecs.transpose()),
        randSymMat,
        1E-4
      );
    });
  });
});
