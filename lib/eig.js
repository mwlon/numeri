const createUtils = require('./createUtils');
const generalUtils = require('./generalUtils');
const { matMulVec, vecMulMat, outerProdInto, matMul } = require('./matrixUtils');

function applySwap(tensor, x0, y0, x1, y1) {
  const val0 = tensor.get(x0, y0);
  tensor.set([x0, y0], tensor.get(x1, y1));
  tensor.set([x1, y1], val0);
}

function applyColumnRot(vecs, i, cos, sin, iterNum) {
  const vec0 = vecs[i];
  const vec1 = vecs[i + 1];
  const n = vecs.length;
  const endJ = Math.min(n, i + 2 + iterNum);
  for (let j = 0; j < endJ; j++) {
    const x = vec0[j];
    const y = vec1[j];
    vec0[j] = cos * x + sin * y;
    vec1[j] = cos * y - sin * x;
  }
}

function applySubmatrixReflection(tensor, startI, axis0, axis1) {
  const n = tensor.shape[0];
  for (let i = startI; i < n; i++) {
    applySwap(tensor, i, axis0, i, axis1);
  }
  for (let i = startI; i < n; i++) {
    applySwap(tensor, axis0, i, axis1, i);
  }
}

function applyColumnSwap(tensor, y0, y1) {
  const n = tensor.shape[0];
  for (let i = 0; i < n; i++) {
    applySwap(tensor, i, y0, i, y1);
  }
}

function tridiagonalQrIter(toDiagonalize, vecs, startI, iterNum) {
  //here each row is treated as vector of 4 elements: 3 on tridiagonal,
  //one more to the right for arithmetic
  //apply givens rotation to zero out subdiagonal
  const n = toDiagonalize.length;
  const coss = Array(n - 1);
  const sins = Array(n - 1);

  const baseRow = toDiagonalize[startI];
  const shiftBTerm = baseRow[2];
  const shift = module.exports.getQrShift(
    baseRow[1],
    shiftBTerm,
    shiftBTerm,
    toDiagonalize[startI + 1][1]
  );

  for (let i = startI; i < n; i++) {
    const row = toDiagonalize[i];
    row[1] -= shift;
  }

  for (let i = startI; i < n - 1; i++) {
    const baseRow = toDiagonalize[i];
    const nextRow = toDiagonalize[i + 1];
    const dx = baseRow[1];
    const dy = nextRow[0];
    const dr = Math.sqrt(dx * dx + dy * dy);
    if (dr === 0) {
      //consecutive identical eigenvalues, already along diagonal
      coss[i] = 1;
      sins[i] = 0;
      continue;
    }
    const cos = dx / dr;
    const sin = dy / dr;

    if (vecs) {
      applyColumnRot(vecs, i, cos, sin, iterNum);
    }

    const nextDx = baseRow[2];
    const nextDy = nextRow[1];
    const lastDy = nextRow[2];

    //two rows affected in 3 columns
    baseRow[1] = dr;
    nextRow[0] = 0;

    baseRow[2] = cos * nextDx + sin * nextDy;
    nextRow[1] = cos * nextDy - sin * nextDx;

    if (i < n - 2) {
      baseRow[3] = sin * lastDy;
      nextRow[2] = cos * lastDy;
    }

    coss[i] = cos;
    sins[i] = sin;
  }

  for (let i = startI; i < n - 1; i++) {
    const cos = coss[i];
    const sin = sins[i];

    if (i > 0) {
      const prevRow = toDiagonalize[i - 1];
      const prevDx = prevRow[2];
      const prevDy = prevRow[3];
      prevRow[2] = cos * prevDx + sin * prevDy;
      prevRow[3] = 0;
    }

    const baseRow = toDiagonalize[i];
    const dx = baseRow[1];
    const dy = baseRow[2];
    baseRow[1] = cos * dx + sin * dy;
    baseRow[2] = cos * dy - sin * dx;

    const nextRow = toDiagonalize[i + 1];
    const nextDx = nextRow[0];
    const nextDy = nextRow[1];
    nextRow[0] = cos * nextDx + sin * nextDy;
    nextRow[1] = cos * nextDy - sin * nextDx;
  }

  for (let i = startI; i < n; i++) {
    const row = toDiagonalize[i];
    row[1] += shift;
  }
}

function getOffDiagDeviation(rowsToDiagonalize, startI) {
  const deviationNumer = rowsToDiagonalize[startI][2];
  if (deviationNumer === 0) {
    return 0;
  }

  const deviationDenom = rowsToDiagonalize[startI][1];
  return Math.abs(deviationNumer / deviationDenom);
}

module.exports = {
  getQrShift(a, b, c, d) {
    const amd = a - d;
    const desc = Math.sqrt(amd * amd + 4 * b * c);
    const firstTerm = a + d;
    return firstTerm > 0 ? (firstTerm - desc) / 2 : (firstTerm + desc) / 2;
  },

  symHessenberg(tensor, options) {
    const { includeQ, stabilityRatio } = generalUtils.defaults(options, {
      includeQ: false,
      stabilityRatio: 1024
    });

    const hessenberg = tensor.copy();
    const n = tensor.shape[0];
    let q;
    if (includeQ) {
      q = createUtils.identity(n);
    }

    //do householder transformations until tridiagonal
    for (let i = 0; i < n - 2; i++) {
      let householder = hessenberg.slice(i, [i + 1, n]);
      let firstElem = householder.get(0);
      const { max, arg } = generalUtils.argmax(householder.map((x) => Math.abs(x)).data);
      if (max > stabilityRatio * Math.abs(firstElem)) {
        //zero out all entries but the arg'th one
        //then do a rotation to put it in the 0th position
        applySubmatrixReflection(
          hessenberg,
          i,
          i + 1,
          i + 1 + arg
        );
        firstElem = max;
        if (includeQ) {
          applyColumnSwap(q, i + 1, i + 1 + arg);
        }
      }
      householder = householder.copy();
      householder.set([0], firstElem + householder.norm() * Math.sign(firstElem));
      const newNorm = householder.norm();
      householder.mapInPlace((x) => x / newNorm);

      const offDiagEntry = hessenberg.get(i + 1, i) -
        2 * householder.get(0) * householder
          .dot(hessenberg.slice([i + 1, n], i));

      //scale by sqrt 2 to avoid O(n^2) multiply by 2 later on
      householder.mapInPlace((x) => x * Math.sqrt(2));
      const submat = hessenberg.slice([i + 1, n], [i + 1, n]);

      //these updates by (x, y) => x - y would be x - 2 * y
      //but we scaled the householder vector
      outerProdInto(
        householder,
        vecMulMat(householder, submat),
        submat,
        (x, y) => x - y
      );

      outerProdInto(
        matMulVec(submat, householder),
        householder,
        submat,
        (x, y) => x - y
      );

      hessenberg.set([i + 1, i], offDiagEntry);
      hessenberg.set([i, i + 1], offDiagEntry);
      hessenberg.slice(i, [i + 2, n]).setAll(0);
      hessenberg.slice([i + 2, n], i).setAll(0);

      if (includeQ) {
        const qSubmat = q.slice([1, n], [i + 1, n]);
        outerProdInto(
          matMulVec(qSubmat, householder),
          householder,
          qSubmat,
          (x, y) => x - y
        );
      }
    }

    if (includeQ) {
      return {hessenberg, q};
    } else {
      return {hessenberg};
    }
  },

  tridiagonalEig(tensor, options) {
    const { includeVecs, maxIter, tolerance } = generalUtils.defaults(options, {
      includeVecs: false,
      maxIter: Infinity,
      tolerance: 1E-8
    });

    const n = tensor.shape[0];
    // const toDiagonalize = tensor.copy();
    const rowsToDiagonalize = [];
    for (let i = 0; i < n; i++) {
      const row = Array(4).fill(0);
      for (let j = Math.max(0, 1 - i); j < Math.min(3, n - i + 1); j++) {
        row[j] = tensor.get(i, i + j - 1);
      }
      rowsToDiagonalize.push(row);
    }
    let vecs;
    if (includeVecs) {
      vecs = Array(n);
      for (let i = 0; i < n; i++) {
        const vec = Array(n).fill(0);
        vec[i] = 1;
        vecs[i] = vec;
      }
      // vecs = createUtils.identity(n);
    }

    let iterNum = 0;
    for (let startI = 0; startI < n - 1; startI++) {
      for (let iter = 0; iter < maxIter && getOffDiagDeviation(rowsToDiagonalize, startI) > tolerance; iter++) {
        tridiagonalQrIter(rowsToDiagonalize, vecs, startI, iterNum);
        iterNum++;
      }
    }

    const vals = createUtils.empty([n]);
    for (let i = 0; i < n; i++) {
      vals.set([i], rowsToDiagonalize[i][1]);
    }

    const result = {vals};
    if (includeVecs) {
      result.vecs = createUtils.fromNested(vecs).transpose();
    }
    return result;
  },

  symEig(tensor, options) {
    const { includeVecs } = generalUtils.defaults(options, {includeVecs: false});
    const { hessenberg, q } = module.exports.symHessenberg(tensor, {includeQ: includeVecs});
    const { vals, vecs } = module.exports.tridiagonalEig(hessenberg, options);

    const result = {vals};
    if (includeVecs) {
      result.vecs = matMul(q, vecs);
    }
    return result;
  },
};
