const Tensor = require('./Tensor');
const tensorUtils = require('./tensorUtils');

module.exports = {
  matMul(mat0, mat1) {
    if (mat1.shape.length === 1) {
      return module.exports.matMulVec(mat0, mat1);
    } else if (mat0.shape.length === 1) {
      return module.exports.vecMulMat(mat0, mat1);
    }

    tensorUtils.checkMatMulShape(mat0.shape, mat1.shape);
    const h = mat0.shape[0];
    const w = mat1.shape[1];
    const innerDim = mat0.shape[1];
    const data = Array(h * w);

    for (let loc0 = 0; loc0 < h; loc0++) {
      const rowOffset = loc0 * w;
      for (let loc1 = 0; loc1 < w; loc1++) {
        let elem = 0;
        for (let k = 0; k < innerDim; k++) {
          elem += mat0.get(loc0, k) * mat1.get(k, loc1);
        }
        data[rowOffset + loc1] = elem;
      }
    }

    return Tensor.fromDataAndShape(data, [h, w]);
  },

  matMulVec(mat, vec) {
    tensorUtils.checkMatMulShape(mat.shape, vec.shape.concat([1]));

    const h = mat.shape[0];
    const innerDim = mat.shape[1];
    const data = Array(h);

    for (let loc0 = 0; loc0 < h; loc0++) {
      let elem = 0;
      for (let k = 0; k < innerDim; k++) {
        elem += mat.get(loc0, k) * vec.get(k);
      }
      data[loc0] = elem;
    }

    return Tensor.fromDataAndShape(data, [h]);
  },

  vecMulMat(vec, mat) {
    tensorUtils.checkMatMulShape([1].concat(vec.shape), mat.shape);

    const innerDim = mat.shape[0];
    const w = mat.shape[1];
    const data = Array(w);

    for (let loc0 = 0; loc0 < w; loc0++) {
      let elem = 0;
      for (let k = 0; k < innerDim; k++) {
        elem += vec.get(k) * mat.get(k, loc0);
      }
      data[loc0] = elem;
    }

    return Tensor.fromDataAndShape(data, [w]);
  },

  outerProd(vec0, vec1) {
    tensorUtils.checkVector(vec0.shape);
    tensorUtils.checkVector(vec1.shape);

    const h = vec0.shape[0];
    const w = vec1.shape[0];
    const data = Array(h * w);

    for (let loc0 = 0; loc0 < h; loc0++) {
      const rowOffset = loc0 * w;
      for (let loc1 = 0; loc1 < w; loc1++) {
        data[rowOffset + loc1] = vec0.get(loc0) * vec1.get(loc1);
      }
    }

    return Tensor.fromDataAndShape(data, [h, w]);
  },

  outerProdInto(vec0, vec1, out, updateFn) {
    tensorUtils.checkVector(vec0.shape);
    tensorUtils.checkVector(vec1.shape);

    const h = vec0.shape[0];
    const w = vec1.shape[0];
    const outData = out.data;

    for (let loc0 = 0; loc0 < h; loc0++) {
      for (let loc1 = 0; loc1 < w; loc1++) {
        const i = out.getI(loc0, loc1);
        outData[i] = updateFn(
          outData[i],
          vec0.get(loc0) * vec1.get(loc1)
        );
      }
    }

    return out;
  },

  dot(vec0, vec1) {
    tensorUtils.checkDotShape(vec0.shape, vec1.shape);
    const n = vec0.shape[0];
    let res = 0;

    for (let i = 0; i < n; i++) {
      res += vec0.get(i) * vec1.get(i);
    }
    
    return res;
  },
};
