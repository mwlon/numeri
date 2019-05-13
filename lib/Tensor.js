const assert = require('assert');
const tensorUtils = require('./tensorUtils');
const generalUtils = require('./generalUtils');

class Tensor {
  constructor(args) {
    this.data = args.data;
    this.shape = args.shape;
    this.offset = args.offset;
    this.strides = args.strides;
    this.mods = args.mods;

    this.dims = args.shape.length;
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

  //check whether data is in standard order that can be traversed by iterating
  //through data (as opposed to sliced up or transposed)
  isSimple() {
    for (var idx = 0; idx < this.dims; idx++) {
      const stride = this.strides[idx];
      const mod = this.mods[idx]

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

  getReindexer() {
    if (this.isSimple()) {
      if (this.offset === 0) {
        return (i) => i;
      } else {
        return (i) => this.offset + i;
      }
    } else {
      const { mods, strides } = tensorUtils.getStridesAndMods(this.shape);

      return (i) => {
        let res = this.offset;

        for (var idx = 0; idx < this.dims; idx++) {
          const ki = Math.floor((i % mods[idx]) / strides[idx]);
          res += this.strides[idx] * ki;
        }

        return res;
      }
    }
  }

  getLength() {
    this.length = generalUtils.prod(this.shape);
    return this.length;
  }

  getI(loc) {
    let i = 0;
    for (var idx = 0; idx < this.dims; idx++) {
      i += loc[idx] * this.strides[idx];
    }
    return i;
  }

  get(loc) {
    return this.data[this.getI(loc)];
  }

  set(loc, val) {
    this.data[this.getI(loc)] = val;
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
        const [ start = 0, end = this.shape[idx], step = 1 ] = spec;

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

  transpose(perm) {
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

  plus(other) {
    return this.elemwise(other, (a, b) => a + b);
  }

  minus(other) {
    return this.elemwise(other, (a, b) => a - b);
  }

  times(other) {
    return this.elemwise(other, (a, b) => a * b);
  }

  div(other) {
    return this.elemwise(other, (a, b) => a / b);
  }

  elemwise(_other, op) {
    let other;
    if (typeof _other === 'number') {
      other = Tensor.fromDataAndShape(_other, []);
    } else if (_other instanceof Tensor) {
      other = _other;
    } else {
      throw new TypeError('Expected either a number or Tensor.');
    }

    this.checkSameShape(other);
    return this.elemwiseUnsafe(other, op);
  }

  elemwiseUnsafe(other, op) {
    const length = this.getLength();
    const newData = Array(length);

    const thisReindexer = this.getReindexer();
    const otherReindexer = other.getReindexer();

    for (var i = 0; i < length; i++) {
      newData[i] = op(
        this.data[thisReindexer(i)],
        other.data[otherReindexer(i)]
      );
    }

    return Tensor.fromDataAndShape(newData, this.shape);
  }

  checkSameShape(other) {
    assert.deepEqual(
      this.shape,
      other.shape,
      `tensors have different shapes: ${this.shape} vs ${other.shape}`
    );
  }
}

module.exports = Tensor;
