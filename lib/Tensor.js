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
    this.getIByJGetter = this.getCurriedIByJGetter();
    const getIByJ = this.getIByJGetter(jStrides, jMods);
    this.getIByJ = getIByJ;
    this.getByJ = (j) => data[getIByJ(j)];

    this.set = (loc, val) => {data[getI(...loc)] = val;};
    this.update = (loc, fn) => {
      const i = getI(...loc);
      const current = data[i];
      data[i] = fn(current);
    };
  }

  //check whether data is in standard order that can be traversed by iterating
  //through data (as opposed to sliced up or transposed)
  isSimple() {
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

  getCurriedIByJGetter() {
    const { offset, strides, dims, shape } = this;

    if (this.isSimple()) {
      if (offset === 0) {
        return () => (j) => j;
      } else {
        return () => (j) => offset + j;
      }
    }

    if (dims === 1) {
      const stride = strides[0];
      return () => (j) => offset + stride * j;
    }

    const fnTerms = [];
    const jStrideArgs = [];
    const jModArgs = [];
    if (offset !== 0) {
      fnTerms.push(offset.toString());
    }

    for (let idx = 0; idx < dims; idx++) {
      const jStride = `jStride${idx}`;
      const jMod = `jMod${idx}`;
      jStrideArgs.push(jStride);
      jModArgs.push(jMod);
      
      if (shape[idx] !== undefined) {
        const stride = strides[idx];
        const floor = `Math.floor((j % ${jMod})/${jStride})`;
        if (stride === 1) {
          fnTerms.push(floor);
        } else {
          fnTerms.push(`${floor}*${stride}`);
        }
      }
    }

    if (fnTerms.length === 0) {
      return () => () => 0;
    }
    const fnText = '(jStrides, jMods) => {' +
      `const [${jStrideArgs}] = jStrides;` +
      `const [${jModArgs}] = jMods;` +
      `return (j)=>${fnTerms.join('+')};` +
      '}';
    return eval(fnText); //eslint-disable-line no-eval
  }

  getIGetter() {
    const { dims, offset, strides } = this;

    const fnArgs = [];
    const fnTerms = [];
    if (offset !== 0) {
      fnTerms.push(offset.toString());
    }

    for (let idx = 0; idx < dims; idx++) {
      const argName = `loc${idx}`;
      fnArgs.push(argName);

      const stride = strides[idx];
      if (stride === 1) {
        fnTerms.push(argName);
      } else if (stride !== 0) {
        fnTerms.push(`${stride}*${argName}`);
      }
    }

    if (fnTerms.length === 0) {
      return () => 0;
    }

    const fnText = `(${fnArgs.join(',')})=>${fnTerms.join('+')}`;
    return eval(fnText); //eslint-disable-line no-eval
  }

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
    const { data, getIByJ } = this;

    for (let j = 0; j < this.getLength(); j++) {
      const i = getIByJ(j);
      data[i] = op(data[i]);
    }

    return this;
  }

  elemwiseUnaryOp(op) {
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

    const thisGetIByJ = this.getIByJGetter(jStrides, jMods);
    const otherGetIByJ = other.getIByJGetter(jStrides, jMods);
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

    const otherGetIByJ = other.getIByJGetter(jStrides, jMods);

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

  lpSum(p) {
    return this.map((x) => Math.pow(Math.abs(x), p)).sum();
  }

  lpNorm(p) {
    return Math.pow(this.lpSum(p), 1 / p);
  }

  norm() {
    return this.lpNorm(2);
  }

  sum() {
    let s = 0;
    const length = this.getLength();
    for (let j = 0; j < length; j++) {
      s += this.getByJ(j);
    }

    return s;
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
