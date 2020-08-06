const { ShapeError } = require('./errors');

module.exports = {
  getStridesAndMods(shape) {
    const dims = shape.length;
    let stride = 1;
    const mods = Array(dims);
    const strides = Array(dims);

    for (let idx = dims - 1; idx >= 0; idx--) {
      strides[idx] = shape[idx] === undefined ? 0 : stride;
      stride *= shape[idx] || 1;
      mods[idx] = stride;
    }

    return {
      strides,
      mods
    };
  },

  shapesEqual(shape0, shape1) {
    const dims = shape0.length;

    if (shape1.length !== dims) {
      return false;
    }

    for (let idx = 0; idx < dims; idx++) {
      if (shape0[idx] !== shape1[idx]) {
        return false;
      }
    }
    return true;
  },

  checkSameShape(shape0, shape1) {
    if (!module.exports.shapesEqual(shape0, shape1)) {
      throw new ShapeError(`Shapes are not identical:\n\t${shape0}\n\t${shape1}`);
    }
  },

  checkShapesBroadcast(shape0, shape1) {
    const msg = `Tensor shapes do not broadcast: ${shape0} vs ${shape1}. Use broadastOn.`;

    if (shape0.length !== shape1.length) {
      throw new ShapeError(msg);
    }

    for (let idx = 0; idx < shape0.length; idx++) {
      const side0 = shape0[idx];
      const side1 = shape1[idx];
      if (side0 !== undefined && side1 !== undefined && side0 !== side1) {
        throw new ShapeError(msg);
      }
    }
  },

  checkShapeBroadcastsTo(shape0, shape1) {
    const msg = `Shape ${shape0} cannot be broadcast to ${shape1}.`;

    if (shape0.length !== shape1.length) {
      throw new ShapeError(msg);
    }

    for (let idx = 0; idx < shape0.length; idx++) {
      const side0 = shape0[idx];
      const side1 = shape1[idx];
      if (side0 !== undefined && side0 !== side1) {
        throw new ShapeError(msg);
      }
    }
  },

  checkDotShape(shape0, shape1) {
    module.exports.checkVector(shape0);
    module.exports.checkVector(shape1);

    if (shape0[0] !== shape1[0]) {
      throw new ShapeError(
        `Cannot perform dot product on mismatched shapes ${shape0}, ${shape1}`
      );
    }
  },

  checkVector(shape) {
    if (shape.length !== 1) {
      throw new ShapeError(`Tensor shape is not a vector: ${shape}`);
    }
  },

  checkMatMulShape(shape0, shape1) {
    for (const shape of [shape0, shape1]) {
      if (shape.length !== 2) {
        throw new ShapeError(
          `Cannot perform matrix multiplication on ${shape.length}-d tensor of shape ${shape}`
        );
      }
    }

    if (shape0[1] !== shape1[0]) {
      throw new ShapeError(
        `Cannot perform matrix multiplication on mismatched shapes ${shape0}, ${shape1}`
      );
    }
  }
};
