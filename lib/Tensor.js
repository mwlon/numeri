const tensorUtils = require('./tensorUtils');
const generalUtils = require('./generalUtils');
const assert = require('assert');
const arithmeticFnBuilder = require('./arithmeticFnBuilder');

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

    this.initialize();
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

  initialize() {
    const { data, shape } = this;
    this.dims = shape.length;

    const {getI, getIByJGetter} = arithmeticFnBuilder.buildTensorGetters(this);
    this.getI = getI;
    this.get = (...args) => data[getI(...args)];
    this.set = (loc, val) => {
      data[getI(...loc)] = val;
    };
    this.update = (loc, fn) => {
      const i = getI(...loc);
      data[i] = fn(data[i]);
    }

    this.getIByJGetter = getIByJGetter;
  }

  maybeInitializeGetIByJ() {
    if (!this.getIByJ) {
      const { data, shape } = this;
      const { strides: jStrides, mods: jMods } = tensorUtils.getStridesAndMods(shape);
      const getIByJ = this.getIByJGetter(jStrides, jMods);
      this.getIByJ = getIByJ;
      this.getByJ = (j) => data[getIByJ(j)];
    }
  }

  //get a function from j => i
  //where i corresponds to this tensor's data size
  //and j corresponds to some broadcasting tensor's canonical index
  getIByJGetterFast(broadcastShape, jStrides, jMods) {
    const { shape } = this;

    if (tensorUtils.shapesEqual(shape, broadcastShape)) {
      this.maybeInitializeGetIByJ();
      return this.getIByJ;
    }

    return this.getIByJGetter(jStrides, jMods);
  }

  //check whether data is in standard order that can be traversed by iterating
  //through data (as opposed to sliced up or transposed)
  isContiguous() {
    for (let idx = 0; idx < this.dims; idx++) {
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

  getLength() {
    return generalUtils.prod(this.shape);
  }

  getLoc(i) {
    const res = Array(this.dims);
    for (let idx = 0; idx < this.dims; idx++) {
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

    for (let idx = 0; idx < this.dims; idx++) {
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

    for (let idx = 0; idx < this.dims; idx++) {
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
    for (let newIdx = 0; newIdx < newDims; newIdx++) {
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
    this.maybeInitializeGetIByJ();
    const { data, getIByJ } = this;

    for (let j = 0; j < this.getLength(); j++) {
      const i = getIByJ(j);
      data[i] = op(data[i]);
    }

    return this;
  }

  elemwiseUnaryOp(op) {
    this.maybeInitializeGetIByJ();
    const { getByJ, shape } = this;
    const length = this.getLength();
    const newData = Array(length);

    for (let j = 0; j < length; j++) {
      newData[j] = op(getByJ(j));
    }

    return Tensor.fromDataAndShape(newData, shape);
  }

  mapInPlace(op) {
    return this.elemwiseUnaryOpInPlace(op);
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

    const thisGetIByJ = this.getIByJGetterFast(resultShape, jStrides, jMods);
    const otherGetIByJ = other.getIByJGetterFast(resultShape, jStrides, jMods);
    const newData = Array(length);

    for (let j = 0; j < length; j++) {
      newData[j] = op(
        this.data[thisGetIByJ(j)],
        other.data[otherGetIByJ(j)]
      );
    }

    return Tensor.fromDataAndShape(newData, resultShape);
  }

  elemwiseBinaryOpInPlace(other, op) {
    const { shape, data } = this;
    tensorUtils.checkShapeBroadcastsTo(other.shape, shape);

    const { strides: jStrides, mods: jMods } = tensorUtils.getStridesAndMods(shape);
    const length = this.getLength();

    const otherGetIByJ = other.getIByJGetterFast(shape, jStrides, jMods);

    for (let j = 0; j < length; j++) {
      const thisI = this.getIByJ(j);
      data[thisI] = op(
        data[thisI],
        other.data[otherGetIByJ(j)]
      );
    }

    return this;
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

  assign(other) {
    return this.elemwiseBinaryOpInPlace(other, (a, b) => b);
  }

  //reductions

  //shouldKeepIndexFn: (accumulator, val) => boolean
  //this keeps -1 as the index whenever it's not found
  reduceSingleAxisToIndex(
    initialValue,
    shouldKeepIndexFn,
    options
  ) {
    const {
      axis,
      keepScalarAsTensor,
      includeValues
    } = generalUtils.defaults(options, {
      axis: this.shape.length - 1,
      keepScalarAsTensor: false,
      includeValues: false,
    });

    this.maybeInitializeGetIByJ();

    const outShape = this.shape.filter((_, idx) => axis !== idx);
    if (outShape.length !== this.shape.length - 1) {
      throw new Error(`Invalid axis ${axis} for shape ${this.shape}`);
    }
    const outLength = generalUtils.prod(outShape);

    const resultData = Array(outLength).fill(-1);
    const values = Array(outLength).fill(initialValue);
    let result = Tensor.fromDataAndShape(resultData, outShape);

    const bcastResult = result.broadcastOn(axis);
    const getResultIByThisJ = bcastResult.getIByJGetterFast(
      this.shape,
      this.strides,
      this.mods
    );

    let axisJGetter;
    if (axis === this.shape.length - 1) {
      const lastShape = this.shape[this.shape.length - 1];
      axisJGetter = (j) => j % lastShape;
    } else {
      const { strides, mods } = tensorUtils.getStridesAndMods(this.shape);
      const axisMod = mods[axis];
      const axisStride = strides[axis];
      axisJGetter = (j) => Math.floor((j % axisMod) / axisStride);
    }

    const getByJ = this.getByJ;
    for (let j = 0; j < this.getLength(); j++) {
      const outI = getResultIByThisJ(j);
      const value = getByJ(j);
      if (shouldKeepIndexFn(values[outI], value)) {
        const axisJ = axisJGetter(j);
        values[outI] = value;
        resultData[outI] = axisJ;
      }
    }

    let valuesResult;

    if (result.shape.length === 0 && !keepScalarAsTensor) {
      result = resultData[0];
      valuesResult = values[0];
    } else if (includeValues) {
      valuesResult = Tensor.fromDataAndShape(values, outShape);
    }

    if (includeValues) {
      return {
        arg: result,
        values: valuesResult
      };
    }

    return result;
  }

  argmax(options) {
    return this.reduceSingleAxisToIndex(
      -Infinity,
      (current, x) => x > current,
      options
    );
  }

  argmin(options) {
    return this.reduceSingleAxisToIndex(
      Infinity,
      (current, x) => x < current,
      options
    );
  }

  //updateFn: (accumulator, val) => newAccumulator
  //finalizeFn: (accumulator) => finalResult
  reduce(
    initialValue,
    updateFn,
    finalizeFn,
    options
  ) {
    const { axes, keepScalarAsTensor } = generalUtils.defaults(options, {
      keepScalarAsTensor: false
    });

    this.maybeInitializeGetIByJ();

    let outShape;
    if (axes === undefined) {
      outShape = [];
    } else {
      outShape = this.shape.filter((_, idx) => axes.indexOf(idx) === -1);
      if (outShape.length !== this.shape.length - axes.length) {
        throw new Error(`Invalid axes ${axes} for shape ${this.shape}`);
      }
    }
    const outLength = generalUtils.prod(outShape);

    const resultData = Array(outLength).fill(initialValue);
    const result = Tensor.fromDataAndShape(resultData, outShape);

    let getResultIByThisJ;
    if (axes === undefined) {
      getResultIByThisJ = () => 0;
    } else {
      const bcastResult = result.broadcastOn(...axes);
      getResultIByThisJ = bcastResult.getIByJGetterFast(
        this.shape,
        this.strides,
        this.mods
      );
    }

    for (let j = 0; j < this.getLength(); j++) {
      const outI = getResultIByThisJ(j);
      resultData[outI] = updateFn(resultData[outI], this.getByJ(j));
    }

    if (finalizeFn !== undefined) {
      for (let i = 0; i < outLength; i++) {
        resultData[i] = finalizeFn(resultData[i]);
      }
    }

    if (result.shape.length === 0 && !keepScalarAsTensor) {
      return result.data[0];
    }

    return result;
  }

  lpNorm(p, options) {
    return this.reduce(
      0,
      (acc, x) => acc + Math.pow(Math.abs(x), p),
      (acc) => Math.pow(acc, 1 / p),
      options
    );
  }

  norm(options) {
    return this.lpNorm(2, options);
  }

  sum(options) {
    return this.reduce(
      0,
      (acc, x) => acc + x,
      undefined,
      options
    );
  }

  max(options) {
    return this.reduce(
      -Infinity,
      (acc, x) => Math.max(acc, x),
      undefined,
      options
    );
  }

  min(options) {
    return this.reduce(
      Infinity,
      (acc, x) => Math.min(acc, x),
      undefined,
      options
    );
  }

  toNested() {
    if (this.dims === 0) {
      return this.get(0);
    } else if (this.dims === 1) {
      return this.copy().data;
    } else {
      const firstDim = this.shape[0];
      const res = Array(firstDim);
      for (let i = 0; i < firstDim; i++) {
        res[i] = this.slice(i).toNested();
      }
      return res;
    }
  }
}

module.exports = Tensor;
