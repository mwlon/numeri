const createUtils = require('./createUtils');
const { ShapeError } = require('./errors');

module.exports = {
  stack: (tensors, axis) => {
    if (tensors.length === 0) {
      throw ShapeError('Cannot stack empty list of tensors');
    }
    const dims = tensors[0].shape.length;
    const finalShape = Array(dims + 1).fill(undefined);
    tensors.forEach((tensor) => {
      if (tensor.shape.length !== dims) {
        throw ShapeError(`Cannot stack ${dims}-dimension tensor with ${tensor.shape.length}-dimensional tensor`);
      }
      tensor.shape.forEach((side, idx) => {
        const finalIdx = idx < axis ? idx : idx + 1;
        if (side !== undefined) {
          if (finalShape[finalIdx] === undefined) {
            finalShape[finalIdx] = side;
          } else if (finalShape[finalIdx] !== side) {
            throw ShapeError(`Tensors have mismatching sides ${finalShape[finalIdx]} and ${side} along axis ${idx}`);
          }
        }
      });
    });

    finalShape[axis] = tensors.length;
    const result = createUtils.empty(finalShape);
    tensors.forEach((tensor, tensorIdx) => {
      const slices = Array(dims + 1).fill(undefined);
      slices[axis] = tensorIdx;
      result
        .slice(...slices)
        .assign(tensor);
    });

    return result;
  },

  concat: (tensors, axis) => {
    if (tensors.length === 0) {
      throw ShapeError('Cannot concat empty list of tensors');
    }
    const dims = tensors[0].shape.length;
    const finalShape = Array(dims).fill(undefined);
    let combinedSide = 0;
    tensors.forEach((tensor) => {
      if (tensor.shape.length !== dims) {
        throw ShapeError(`Cannot concat ${dims}-dimension tensor with ${tensor.shape.length}-dimensional tensor`);
      }
      tensor.shape.forEach((side, idx) => {
        if (idx === axis) {
          if (side === undefined) {
            throw ShapeError('Cannot concat tensors that are broadcasted along combination axis');
          }
          combinedSide += side;
        } else if (side !== undefined) {
          if (finalShape[idx] === undefined) {
            finalShape[idx] = side;
          } else if (finalShape[idx] !== side) {
            throw ShapeError(`Tensors have mismatching sides ${finalShape[idx]} and ${side} along axis ${idx}`);
          }
        }
      });
    });

    finalShape[axis] = combinedSide;
    const result = createUtils.empty(finalShape);
    let axisOffset = 0;
    tensors.forEach((tensor) => {
      const slices = Array(dims).fill(undefined);
      slices[axis] = [axisOffset, axisOffset + tensor.shape[axis]];
      result
        .slice(...slices)
        .assign(tensor);
      axisOffset += tensor.shape[axis];
    });

    return result;
  }
};
