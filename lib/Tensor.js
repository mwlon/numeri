const tensorUtils = require('./tensorUtils');
const generalUtils = require('./generalUtils');
const assert = require('assert');

//here `i` refers to index into the `data`, whereas
//`j` refers to canonical index of the element, as if
//this had default strides and 0 offset

class Tensor {
  constructor(args) {
    this.data = args.data;
    this.shape = args.shape;
    this.offset = args.offset;
    this.strides = args.strides;
    this.mods = args.mods;

    this.dims = args.shape.length;
    this.setup();
  }

  static fromDataAndShape(data, shape) {
    const { strides, mods } = tensorUtils.getStridesAndMods(shape);

    return new Tensor({
      data,
      shape,
      offset: 0,
      strides,
      mods
    });
  }

  setup() {
    const { data } = this;

    const getI = this.getIGetter();
    this.getI = getI;
    this.get = (...args) => data[getI(...args)];

    const { strides: jStrides, mods: jMods } = tensorUtils.getStridesAndMods(this.shape);
    const getIByJ = this.getIByJGetter(jStrides, jMods);
    this.getIByJ = getIByJ;
    this.getByJ = (j) => data[getIByJ(j)];

    this.set = (loc, val) => {data[getI(...loc)] = val;};
  }

  //check whether data is in standard order that can be traversed by iterating
  //through data (as opposed to sliced up or transposed)
  isSimple() {
    for (var idx = 0; idx < this.dims; idx++) {
      const stride = this.strides[idx];
      const mod = this.mods[idx];

      if (idx === this.dims - 1) {
        if (stride !== 1) {
          return false;
        }
      } else if (stride !== this.mods[idx + 1]) {
        return false;
      }

      if (mod !== this.strides[idx] * this.shape[idx]) {
        return false;
      }
    }

    return true;
  }

  //element access methods

  getIByJGetter(jStrides, jMods) {
    const { offset, strides, dims, shape } = this;

    if (this.isSimple()) {
      if (offset === 0) {
        return (j) => j;
      } else {
        return (j) => offset + j;
      }
    }

    if (dims === 1) {
      const stride = strides[0];
      return (j) => offset + stride * j;
    }

    const fns = [() => offset];
    for (var idx = 0; idx < dims; idx++) {
      if (jStrides[idx] !== 0 && shape[idx] !== undefined) {
        const jMod = jMods[idx];
        const jStride = jStrides[idx];
        const stride = strides[idx];
        const lastFn = fns[fns.length - 1];
        fns.push((j) => lastFn(j) + Math.floor((j % jMod) / jStride) * stride);
      }
    }

    return fns[fns.length - 1];
    // if (this.isSimple()) {
    //   if (offset === 0) {
    //     return (j) => j;
    //   } else {
    //     return (j) => offset + j;
    //   }
    // } else if (dims === 2) {
    //   const [ stride0, stride1 ] = strides;
    //   const [ jStride0, jStride1 ] = jStrides;
    //   const [ jMod0, jMod1 ] = jMods;

    //   return (j) => offset +
    //       Math.floor((j % jMod0) / jStride0) * stride0 +
    //       Math.floor((j % jMod1) / jStride1) * stride1;
    // } else {
    //   return (j) => {
    //     let res = offset;

    //     for (var idx = 0; idx < dims; idx++) {
    //       const locIdx = Math.floor((j % jMods[idx]) / jStrides[idx]);
    //       res += strides[idx] * locIdx;
    //     }

    //     return res;
    //   };
    // }
  }

  getIGetter() {
    const { dims, offset, strides } = this;

    if (dims === 0) {
      const res = offset;
      return () => res;
    } else if (dims === 1) {
      const stride0 = strides[0];

      if (offset === 0) {
        if (stride0 === 1) {
          return (loc0) => loc0;
        } else {
          return (loc0) => stride0 * loc0;
        }
      } else {
        return (loc0) => offset + stride0 * loc0;
      }
    } else if (dims === 2) {
      const [ stride0, stride1 ] = strides;
      if (offset === 0) {
        if (stride1 === 1) {
          return (loc0, loc1) => stride0 * loc0 + loc1;
        } else {
          return (loc0, loc1) => stride0 * loc0 + stride1 * loc1;
        }
      } else {
        return (loc0, loc1) => offset + stride0 * loc0 + stride1 * loc1;
      }
    } else {
      //not really any specialized ops for these, so no need to unroll
      return (...loc) => {
        let i = offset;
        for (var idx = 0; idx < dims; idx++) {
          i += loc[idx] * strides[idx];
        }
        return i;
      };
    }
  }

  getLength() {
    return generalUtils.prod(this.shape);
  }

  getLoc(i) {
    const res = Array(this.dims);
    for (var idx = 0; idx < this.dims; idx++) {
      res[idx] = Math.floor((i % this.mods[idx]) / this.strides[idx]);
    }
    return res;
  }

  setAll(val) {
    this.elemwiseUnaryOpInPlace(() => val);
  }

  slice(...specs) {
    const shape = [];
    const strides = [];
    const mods = [];
    let offset = this.offset;

    for (var idx = 0; idx < this.dims; idx++) {
      const spec = specs[idx];
      if (spec === undefined) {
        //keep index entirely
        shape.push(this.shape[idx]);
        strides.push(this.strides[idx]);
        mods.push(this.mods[idx]);

      } else if (typeof spec === 'number') {
        //eliminate this index
        offset += this.strides[idx] * spec;
      } else if (Array.isArray(spec)) {
        const [ looseStart = 0, looseEnd = this.shape[idx], step = 1 ] = spec;
        const start = Math.max(looseStart, 0);
        const end = Math.min(looseEnd, this.shape[idx]);

        offset += this.strides[idx] * start;
        shape.push(Math.ceil((end - start) / step));
        strides.push(this.strides[idx] * step);
        mods.push(this.mods[idx]);
      }
    }

    return new Tensor({
      data: this.data,
      shape,
      offset,
      strides,
      mods
    });
  }

  transpose(_perm) {
    let perm = _perm;
    if (this.dims === 2 && !_perm) {
      perm = [1, 0];
    }

    assert.strictEqual(
      this.dims,
      perm.length,
      `Transpose permutation length (${perm.length}) does not match tensor dimensionality (${this.dims}).`
    );
    const shape = Array(this.dims);
    const strides = Array(this.dims);
    const mods = Array(this.dims);

    for (var idx = 0; idx < this.dims; idx++) {
      const permIdx = perm[idx];
      shape[permIdx] = this.shape[idx];
      strides[permIdx] = this.strides[idx];
      mods[permIdx] = this.mods[idx];
    }

    return new Tensor({
      data: this.data,
      shape,
      offset: this.offset,
      strides,
      mods
    });
  }

  broadcastOn(...dimList) {
    const dimSet = new Set(dimList);

    const newDims = this.dims + dimList.length;
    const newShape = Array(newDims);
    const newStrides = Array(newDims);
    const newMods = Array(newDims);

    let idx = 0;
    for (var newIdx = 0; newIdx < newDims; newIdx++) {
      if (dimSet.has(newIdx)) {
        newShape[newIdx] = undefined;
        newStrides[newIdx] = 0;
        newMods[newIdx] = 1;
      } else {
        newShape[newIdx] = this.shape[idx];
        newStrides[newIdx] = this.strides[idx];
        newMods[newIdx] = this.mods[idx];
        idx ++;
      }
    }

    return new Tensor({
      data: this.data,
      shape: newShape,
      offset: this.offset,
      strides: newStrides,
      mods: newMods
    });
  }

  reshape(shape) {
    assert.strictEqual(
      generalUtils.prod(shape),
      generalUtils.prod(this.shape),
      `Unable to reshape ${this.shape} tensor into shape ${shape}`
    );

    const data = this.copy().data;
    return Tensor.fromDataAndShape(data, shape);
  }

  //unary operators

  elemwiseUnaryOpInPlace(op) {
    const { data, getIByJ } = this;

    for (var j = 0; j < this.getLength(); j++) {
      const i = getIByJ(j);
      data[i] = op(data[i]);
    }

    return this;
  }

  elemwiseUnaryOp(op) {
    const { getByJ, shape } = this;
    const length = this.getLength();
    const newData = Array(length);

    for (var j = 0; j < length; j++) {
      newData[j] = op(getByJ(j));
    }

    return Tensor.fromDataAndShape(newData, shape);
  }

  map(op) {
    return this.elemwiseUnaryOp(op);
  }

  copy() {
    return this.elemwiseUnaryOp((x) => x);
  }

  exp() {
    return this.elemwiseUnaryOp((x) => Math.exp(x));
  }

  //binary operators

  elemwiseBinaryOp(_other, op) {
    let other;
    if (typeof _other === 'number') {
      other = Tensor.fromDataAndShape([_other], this.shape.map(() => undefined));
    } else if (_other instanceof Tensor) {
      tensorUtils.checkShapesBroadcast(this.shape, _other.shape);
      other = _other;
    } else {
      throw new TypeError('Binary op expected either a number or Tensor.');
    }

    return this.elemwiseBinaryOpUnsafe(other, op);
  }

  elemwiseBinaryOpUnsafe(other, op) {
    const resultShape = this.shape.map((side, idx) =>
      side === undefined ? other.shape[idx] : side
    );
    const { strides: jStrides, mods: jMods } = tensorUtils.getStridesAndMods(resultShape);
    const length = generalUtils.prod(resultShape);

    const thisGetIByJ = this.getIByJGetter(jStrides, jMods);
    const otherGetIByJ = other.getIByJGetter(jStrides, jMods);
    const newData = Array(length);

    for (var j = 0; j < length; j++) {
      newData[j] = op(
        this.data[thisGetIByJ(j)],
        other.data[otherGetIByJ(j)]
      );
    }

    return Tensor.fromDataAndShape(newData, resultShape);
  }

  plus(other) {
    return this.elemwiseBinaryOp(other, (a, b) => a + b);
  }

  minus(other) {
    return this.elemwiseBinaryOp(other, (a, b) => a - b);
  }

  times(other) {
    return this.elemwiseBinaryOp(other, (a, b) => a * b);
  }

  div(other) {
    return this.elemwiseBinaryOp(other, (a, b) => a / b);
  }

  matMul(other) {
    tensorUtils.checkMatMulShape(this.shape, other.shape);

    const data = Array(this.shape[0] * other.shape[1]).fill(0);

    for (var loc0 = 0; loc0 < this.shape[0]; loc0++) {
      const rowOffset = loc0 * other.shape[1];
      for (var loc1 = 0; loc1 < other.shape[1]; loc1++) {
        let elem = 0;
        for (var k = 0; k < this.shape[1]; k++) {
          elem += this.get(loc0, k) * other.get(k, loc1);
        }
        data[rowOffset + loc1] = elem;
      }
    }

    return Tensor.fromDataAndShape(data, [this.shape[0], other.shape[1]]);
  }
}

module.exports = Tensor;
