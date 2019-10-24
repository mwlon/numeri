const tensorUtils = require('./tensorUtils');
const generalUtils = require('./generalUtils');
const assert = require('assert');

//here `i` refers to index into the `data`, whereas
//`j` refers to canonical index of the element, as if
//this had default strides and 0 offset

const iTermFn = (stride) => {
  if (stride === 1) {
    return (loc) => loc;
  } else if (stride !== 0) {
    return (loc) => loc * stride;
  } else {
    return () => 0;
  }
};

const iByJTermFn = (jStride, jMod, stride, shape) => {
  if (shape === undefined) {
    return () => 0;
  } else {
    return (j) => Math.floor((j % jMod) / jStride) * stride;
  }
};

class Tensor {
  constructor(args) {
    this.data = args.data;
    this.shape = args.shape;
    this.offset = args.offset;
    this.strides = args.strides;
    this.mods = args.mods;

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
    const { data, shape } = this;
    this.dims = shape.length;

    const getI = this.getIGetter();
    this.getI = getI;
    this.get = (...args) => data[getI(...args)];

    const { strides: jStrides, mods: jMods } = tensorUtils.getStridesAndMods(shape);
    const getIByJGetter = this.getCurriedIByJGetter();
    this.getIByJGetter = getIByJGetter;
    const getIByJ = getIByJGetter(jStrides, jMods);
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

  getCurriedIByJGetter() {
    const { offset, strides, dims, shape } = this;

    //make this function run faster by handling most cases without eval
    if (this.isContiguous()) {
      if (offset === 0) {
        return () => (j) => j;
      } else {
        return ((o) => () => (j) => o + j)(offset);
      }
    } else if (dims === 1) {
      const stride = strides[0];
      if (offset === 0) {
        return ((s) => () => (j) => s * j)(stride);
      } else {
        return ((o, s) => () => (j) => o + s * j)(offset, stride);
      }
    } else if (dims === 2) {
      const [ stride0, stride1 ] = strides;
      const [ shape0, shape1 ] = shape;
      return (jStrides, jMods) => {
        const [ jStride0, jStride1 ] = jStrides;
        const [ jMod0, jMod1 ] = jMods;
        const term0Fn = iByJTermFn(jStride0, jMod0, stride0, shape0);
        const term1Fn = iByJTermFn(jStride1, jMod1, stride1, shape1);
        if (offset === 0) {
          return (j) => term0Fn(j) + term1Fn(j);
        } else {
          return ((o) => (j) => term0Fn(j) + term1Fn(j) + o)(offset);
        }
      };
    }

    //general case with eval
    const fnTerms = [];
    const jStrideArgs = Array(dims);
    const jModArgs = Array(dims);
    if (offset !== 0) {
      fnTerms.push(offset.toString());
    }

    for (let idx = 0; idx < dims; idx++) {
      const jStride = `jStride${idx}`;
      const jMod = `jMod${idx}`;
      jStrideArgs[idx] = jStride;
      jModArgs[idx] = jMod;

      if (shape[idx] !== undefined) {
        const stride = strides[idx];
        const floor = `Math.floor((j % ${jMod}) / ${jStride})`;
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
    const fnText = '(jStrides,jMods)=>{' +
      `const [${jStrideArgs.join(',')}]=jStrides;` +
      `const [${jModArgs.join(',')}]=jMods;` +
      `return (j)=>${fnTerms.join('+')};` +
      '}';
    return eval(fnText); //eslint-disable-line no-eval
  }

  getIGetter() {
    const { dims, offset, strides } = this;

    if (dims === 0) {
      return ((o) => () => o)(offset);
    } else if (dims === 1) {
      const stride = strides[0];
      const termFn = iTermFn(stride);
      if (offset === 0) {
        return termFn;
      } else {
        return ((o) => (loc0) => termFn(loc0) + o)(offset);
      }
    } else if (dims === 2) {
      const [ stride0, stride1 ] = strides;
      const term0Fn = iTermFn(stride0);
      const term1Fn = iTermFn(stride1);
      if (offset === 0) {
        return (loc0, loc1) => term0Fn(loc0) + term1Fn(loc1);
      } else {
        return ((o) => (loc0, loc1) => term0Fn(loc0) + term1Fn(loc1) + o)(offset);
      }
    }

    const fnArgs = Array(dims);
    const fnTerms = [];
    if (offset !== 0) {
      fnTerms.push(offset.toString());
    }

    for (let idx = 0; idx < dims; idx++) {
      const argName = `loc${idx}`;
      fnArgs[idx] = argName;

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
