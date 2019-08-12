const createUtils = require('./createUtils');
const generalUtils = require('./generalUtils');

function applySwap(tensor, x0, y0, x1, y1) {
  const val0 = tensor.get(x0, y0);
  tensor.set([x0, y0], tensor.get(x1, y1));
  tensor.set([x1, y1], val0);
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

function tridiagonalQrIter(toDiagonalize, vecs, startI) {
  //apply givens rotation to zero out subdiagonal
  const n = toDiagonalize.shape[0];
  const coss = Array(n - 1);
  const sins = Array(n - 1);

  const shiftBTerm = toDiagonalize.get(startI, startI + 1);
  const shift = module.exports.getQrShift(
    toDiagonalize.get(startI, startI),
    shiftBTerm,
    shiftBTerm,
    toDiagonalize.get(startI + 1, startI + 1)
  );

  for (let i = startI; i < n; i++) {
    toDiagonalize.update([i, i], (x) => x - shift);
  }

  for (let i = startI; i < n - 1; i++) {
    const dx = toDiagonalize.get(i, i);
    const dy = toDiagonalize.get(i + 1, i);
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
      applyColumnRot(vecs, i, cos, sin)
    }

    const nextDx = toDiagonalize.get(i, i + 1);
    const nextDy = toDiagonalize.get(i + 1, i + 1);
    const lastDy = toDiagonalize.get(i + 1, i + 2);

    //two rows affected in 3 columns
    toDiagonalize.set([i, i], dr);
    toDiagonalize.set([i + 1, i], 0);

    toDiagonalize.set(
      [i, i + 1],
      cos * nextDx + sin * nextDy
    );
    toDiagonalize.set(
      [i + 1, i + 1],
      cos * nextDy - sin * nextDx
    );

    if (i < n - 2) {
      toDiagonalize.set(
        [i, i + 2],
        sin * lastDy
      );
      toDiagonalize.set(
        [i + 1, i + 2],
        cos * lastDy
      );
    }

    coss[i] = cos;
    sins[i] = sin;
  }

  for (let i = startI; i < n - 1; i++) {
    const cos = coss[i];
    const sin = sins[i];

    if (i > 0) {
      const prevDx = toDiagonalize.get(i - 1, i);
      const prevDy = toDiagonalize.get(i - 1, i + 1);
      toDiagonalize.set(
        [i - 1, i],
        cos * prevDx + sin * prevDy
      );
      toDiagonalize.set(
        [i - 1, i + 1],
        0
      );
    }

    const dx = toDiagonalize.get(i, i);
    const dy = toDiagonalize.get(i, i + 1);
    toDiagonalize.set(
      [i, i],
      cos * dx + sin * dy
    );
    toDiagonalize.set(
      [i, i + 1],
      cos * dy - sin * dx
    );

    const nextDx = toDiagonalize.get(i + 1, i);
    const nextDy = toDiagonalize.get(i + 1, i + 1);
    toDiagonalize.set(
      [i + 1, i],
      cos * nextDx + sin * nextDy
    );
    toDiagonalize.set(
      [i + 1, i + 1],
      cos * nextDy - sin * nextDx
    );
  }

  for (let i = startI; i < n; i++) {
    toDiagonalize.update([i, i], (x) => x + shift);
  }
}

function getOffDiagDeviation(toDiagonalize, startI) {
  const deviationNumer = toDiagonalize.get(startI, startI + 1);
  if (deviationNumer === 0) {
    return 0;
  }

  const deviationDenom = toDiagonalize.get(startI, startI);
  return Math.abs(deviationNumer / deviationDenom);
}

function applyColumnRot(vecs, i, cos, sin) {
  const n = vecs.shape[0];
  for (let j = 0; j < n; j++) {
    const x = vecs.get(j, i);
    const y = vecs.get(j, i + 1);
    vecs.set([j, i], cos * x + sin * y);
    vecs.set([j, i + 1], cos * y - sin * x);
  }
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
      includeQ: true,
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
      const vCol = householder.reshape([n - i - 1, 1]);
      const vRow = householder.reshape([1, n - i - 1]);

      const offDiagEntry = hessenberg.get(i + 1, i) -
        2 * householder.get(0) * householder
          .dot(hessenberg.slice([i + 1, n], i));
      const submat = hessenberg.slice([i + 1, n], [i + 1, n]);

      const rowAdjustment = vCol
        .matMul(vRow
          .matMul(submat)
        );
      submat.elemwiseBinaryOpInPlace(
        rowAdjustment,
        (x, y) => x - 2 * y
      );

      const colAdjustment = submat
        .matMul(vCol)
        .matMul(vRow);
      submat.elemwiseBinaryOpInPlace(
        colAdjustment,
        (x, y) => x - 2 * y
      );

      hessenberg.set([i + 1, i], offDiagEntry);
      hessenberg.set([i, i + 1], offDiagEntry);
      hessenberg.slice(i, [i + 2, n]).setAll(0);
      hessenberg.slice([i + 2, n], i).setAll(0);

      if (includeQ) {
        const qSubmat = q.slice([1, n], [i + 1, n]);
        const qAdjustment = qSubmat
          .matMul(vCol)
          .matMul(vRow);
        qSubmat.elemwiseBinaryOpInPlace(
          qAdjustment,
          (x, y) => x - 2 * y
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
      includeVecs: true,
      maxIter: Infinity,
      tolerance: 1E-12
    });

    const n = tensor.shape[0];
    const toDiagonalize = tensor.copy();
    let vecs;
    if (includeVecs) {
      vecs = createUtils.identity(n);
    }

    let iters = 0;

    for (let startI = 0; startI < n - 1; startI++) {
      for (let iter = 0; iter < maxIter && getOffDiagDeviation(toDiagonalize, startI) > tolerance; iter++) {
        iters += 1;
        tridiagonalQrIter(toDiagonalize, vecs, startI);
        // console.log(startI, iter, getOffDiagDeviation(toDiagonalize, startI), iters);
      }
    }

    const vals = createUtils.empty([n]);
    for (let i = 0; i < n; i++) {
      vals.set([i], toDiagonalize.get(i, i));
    }

    const result = {vals};
    if (includeVecs) {
      result.vecs = vecs;
    }
    return result;
  },

  symEig(tensor, options) {
    const { includeVecs } = generalUtils.defaults(options, {includeVecs: true});
    const { hessenberg, q } = module.exports.symHessenberg(tensor, {includeQ: includeVecs});
    const { vals, vecs } = module.exports.tridiagonalEig(hessenberg, options);

    const result = {vals};
    if (includeVecs) {
      result.vecs = q.matMul(vecs);
    }
    return result;
  },
};
