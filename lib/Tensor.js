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
      };
    }
  }

  getLength() {
    this.length = generalUtils.prod(this.shape);
    return this.length;
  }

  getDataI(loc) {
    let i = this.offset;
    for (var idx = 0; idx < this.dims; idx++) {
      i += loc[idx] * this.strides[idx];
    }
    return i;
  }

  getLoc(i) {
    const res = Array(this.dims);
    for (var idx = 0; idx < this.dims; idx++) {
      res[idx] = Math.floor((i % this.mods[idx]) / this.strides[idx]);
    }
    return res;
  }

  get(loc) {
    return this.data[this.getDataI(loc)];
  }

  set(loc, val) {
    this.data[this.getDataI(loc)] = val;
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

  //unary operators

  elemwiseUnaryOp(op) {
    const data = Array(this.getLength());
    const reindexer = this.getReindexer();

    for (var i = 0; i < this.length; i++) {
      data[i] = op(this.data[reindexer(i)]);
    }

    return Tensor.fromDataAndShape(data, this.shape);
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
      other = Tensor.fromDataAndShape(_other, []);
    } else if (_other instanceof Tensor) {
      other = _other;
    } else {
      throw new TypeError('Binary op expected either a number or Tensor.');
    }

    tensorUtils.checkSameShape(this.shape, other.shape);
    return this.elemwiseBinaryOpUnsafe(other, op);
  }

  elemwiseBinaryOpUnsafe(other, op) {
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

    for (var i = 0; i < this.shape[0]; i++) {
      for (var j = 0; j < other.shape[1]; j++) {
        for (var k = 0; k < this.shape[1]; k++) {
          data[i * other.shape[1] + j] += this.get([i, k]) * other.get([k, j]);
        }
      }
    }

    return Tensor.fromDataAndShape(data, [this.shape[0], other.shape[1]]);
  }

  //output

  toNested() {

  }
}

module.exports = Tensor;
