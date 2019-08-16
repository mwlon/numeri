const assert = require('assert');

module.exports = {
  getStridesAndMods(shape) {
    const dims = shape.length;
    let stride = 1;
    const mods = Array(dims);
    const strides = Array(dims);

    for (var idx = dims - 1; idx >= 0; idx--) {
      strides[idx] = shape[idx] === undefined ? 0 : stride;
      stride *= shape[idx] || 1;
      mods[idx] = stride;
    }

    return {
      strides,
      mods
    };
  },

  checkSameShape(shape0, shape1) {
    assert.deepEqual(
      shape0,
      shape1,
      `Tensors have different shapes: ${shape0} vs ${shape1}`
    );
  },

  checkShapesBroadcast(shape0, shape1) {
    const msg = `Tensor shapes do not broadcast: ${shape0} vs ${shape1}. Use broadastOn.`;

    assert.strictEqual(
      shape0.length,
      shape1.length,
      msg
    );

    for (var idx = 0; idx < shape0.length; idx++) {
      const side0 = shape0[idx];
      const side1 = shape1[idx];
      assert(
        side0 === undefined || side1 === undefined || side0 === side1,
        msg
      );
    }
  },

  checkShapeBroadcastsTo(shape0, shape1) {
    const msg = `Shape ${shape0} cannot be broadcast to ${shape1}.`;

    assert.strictEqual(
      shape0.length,
      shape1.length,
      msg
    );

    for (var idx = 0; idx < shape0.length; idx++) {
      const side0 = shape0[idx];
      const side1 = shape1[idx];
      assert(
        side0 === undefined || side0 === side1,
        msg
      );
    }
  },

  checkDotShape(shape0, shape1) {
    for (var shape of [shape0, shape1]) {
      assert.strictEqual(
        shape.length,
        1,
        `Cannot perform dot product on ${shape.length}-d tensor of shape ${shape}`
      );
    }

    assert.strictEqual(
      shape0[0],
      shape1[0],
      `Cannot perform dot product on mismatched shapes ${shape0}, ${shape1}`
    );
  },

  checkVector(shape) {
    assert.strictEqual(shape.length, 1, `Tensor shape is not a vector: ${shape}`);
  },

  checkMatMulShape(shape0, shape1) {
    for (var shape of [shape0, shape1]) {
      assert.strictEqual(
        shape.length,
        2,
        `Cannot perform matrix multiplication on ${shape.length}-d tensor of shape ${shape}`
      );
    }

    assert.strictEqual(
      shape0[1],
      shape1[0],
      `Cannot perform matrix multiplication on mismatched shapes ${shape0}, ${shape1}`
    );
  },

  getLocByJ(j, strides, mods) {
    const dims = strides.length;
    const res = Array(dims);
    for (var idx = 0; idx < dims; idx++) {
      res[idx] = Math.floor((j % mods[idx]) / strides[idx]);
    }

    return res;
  }
};
