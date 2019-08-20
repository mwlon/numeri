const assert = require('assert');
const generalUtils = require('./generalUtils');
const Tensor = require('./Tensor');
const tensorUtils = require('./tensorUtils');

module.exports = {
  scalar(data) {
    return Tensor.fromDataAndShape([data], []);
  },

  vector(data) {
    return Tensor.fromDataAndShape(data, [data.length]);
  },

  range(n) {
    const data = Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = i;
    }
    return Tensor.fromDataAndShape(data, [n]);
  },

  fromFlat(data, shape) {
    assert.strictEqual(
      data.length,
      generalUtils.prod(shape),
      `Data length ${data.length} did not agree with tensor shape ${shape}.`
    );

    return Tensor.fromDataAndShape(data, shape);
  },

  fromNested(nested) {
    const shape = module.exports.getNestedShape(nested);
    const data = module.exports.getNestedData(nested, shape);
    return Tensor.fromDataAndShape(data, shape);
  },

  getNestedShape(nested, existingShape=[]) {
    if (typeof nested === 'number') {
      return existingShape;
    } else if (Array.isArray(nested)) {
      existingShape.push(nested.length);
      if (nested.length === 0) {
        return existingShape;
      }
      return module.exports.getNestedShape(nested[0], existingShape);
    } else {
      throw new Error(`Unknown data type found in provided nested data: ${nested}`);
    }
  },

  getNestedData(nested, shape) {
    const res = Array(generalUtils.prod(shape));
    const { strides } = tensorUtils.getStridesAndMods(shape);
    module.exports.getNestedDataHelper(nested, shape, res, 0, 0, strides);
    return res;
  },

  getNestedDataHelper(nested, shape, res, idx, offset, strides) {
    if (idx === shape.length) {
      generalUtils.checkNumber(nested);
      res[offset] = nested;
    }

    const n = shape[idx];
    if (nested.length !== n) {
      throw new Error(`Irregular shape ${nested.length} found in nested ` +
        `data at index ${idx}; expected ${n}.`);
    }

    if (idx === shape.length - 1) {
      for (let i = 0; i < n; i++) {
        res[offset + i] = nested[i];
      }
    } else {
      //recurse
      for (let i = 0; i < n; i++) {
        module.exports.getNestedDataHelper(nested[i], shape, res, idx + 1, offset + i * strides[idx], strides);
      }
    }
  },

  fill(shape, val) {
    const data = Array(generalUtils.prod(shape)).fill(val);
    return Tensor.fromDataAndShape(data, shape);
  },

  zeros(shape) {
    return module.exports.fill(shape, 0);
  },

  identity(n) {
    const data = Array(n * n).fill(0);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1;
    }
    return Tensor.fromDataAndShape(data, [n, n]);
  },

  empty(shape) {
    const data = Array(generalUtils.prod(shape));
    return Tensor.fromDataAndShape(data, shape);
  },

  diagonal(entries) {
    const n = entries.length;
    const result = module.exports.zeros([n, n]);
    for (let i = 0; i < n; i++) {
      result.set([i, i], entries[i]);
    }
    return result;
  },
};
