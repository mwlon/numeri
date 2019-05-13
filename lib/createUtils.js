const assert = require('assert');
const generalUtils = require('./generalUtils');
const tensorUtils = require('./tensorUtils');
const Tensor = require('./Tensor');

module.exports = {
  scalar(data) {
    return Tensor.fromDataAndShape(data, []);
  },

  vector(data) {
    return Tensor.fromDataAndShape(data, [data.length]);
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
    const shape = module.exports.getNestedShape(nested, []);
    const data = module.exports.getNestedData(nested, shape, 0);
    return fromDataAndShape(data, shape);
  },

  getNestedShape(nested, existingShape) {
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

  getNestedData(nested, shape, idx) {
    if (idx === shape.length) {
      if (typeof nested !== 'number') {
        throw new Error(`Unknown data type found in provided nested data: ${nested}`);
      }
      return [nested];
    }
    if (nested.length !== shape[idx]) {
      throw new Error(`Irregular shape found in nested data.`);
    }

    if (idx === shape.length - 1) {
      return nested;
    }

    const nestedParts = nested.map((part) => getNestedData(part, shape, idx + 1));
    const partSize = nestedParts[0].length;
    const res = Array(nestedParts.length * partSize);
    for (var i = 0; i < shape[idx]; i++) {
      for (var j = 0; j < partSize; j++) {
        res[i * partSize + j] = nestedParts[i][j];
      }
    }
    
    return res;
  },

  zeros(shape) {
    const data = Array(generalUtils.prod(shape)).fill(0);
    return Tensor.fromDataAndShape(data, shape);
  }
};