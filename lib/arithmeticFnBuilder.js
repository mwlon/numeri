//ok this takes some explanation
//I need to dynamically produce the getter functions for tensors
//because
//  return loc0 + stride1 * loc1;
//is much better than
//  let s = 0; for (let idx = 0; idx < dims; idx++) {s += loc[idx] * stride[idx]}; return s;
//This object helps create efficient functions on the fly, and avoid creating
//similar functions twice, since each one can require milliseconds to format
//and eval.
const generalUtils = require('./generalUtils');

const cache = {};

const getTensorCase = (tensor) => {
  const hashModulus = 4;
  const { strides, offset, shape } = tensor;
  let res = 0;
  for (let idx = 0; idx < strides.length; idx++) {
    const stride = strides[idx];
    const side = shape[idx];
    if (stride === 0 || side === undefined || side <= 1) {
      res += 1;
    } else if (stride === 1) {
      res += 2;
    } else {
      res += 3;
    }
    res *= hashModulus;
  }
  res += (offset !== 0) + 1;
  res *= hashModulus;
  res += (tensor.isContiguous()) + 1;
  return res;
};

const getTensorNumbers = (tensor) => {
  const { strides, offset } = tensor;
  const dims = strides.length;

  const res = Array(dims + 1);
  res[dims] = offset;
  for (let idx = 0; idx < dims; idx++) {
    res[idx] = strides[idx];
  }
  return res;
};

const getCachedGetI = (tensor) => {
  const { strides, offset, shape, dims } = tensor;

  //eliminate security concerns of `new Function`
  generalUtils.checkNumbers(strides);
  generalUtils.checkNumber(offset);

  const fnArgs = Array(dims);
  const getITerms = [];
  const numArgs = Array(dims + 1);
  numArgs[dims] = 'o';
  if (offset !== 0) {
    getITerms.push('o');
  }

  for (let idx = 0; idx < dims; idx++) {
    const argName = `l${idx}`;
    fnArgs[idx] = argName;
    const numArg = `n${idx}`;
    numArgs[idx] = numArg;

    const stride = strides[idx];
    if (stride !== 0 && shape[idx] !== undefined && shape[idx] > 1) {
      if (stride === 1) {
        getITerms.push(argName);
      } else {
        getITerms.push(`${numArg}*${argName}`);
      }
    }
  }

  if (getITerms.length === 0) {
    return () => () => 0;
  }

  return new Function( //eslint-disable-line no-new-func
    ...numArgs,
    `return (${fnArgs.join(',')})=>` +
      `${getITerms.join('+')}`
  );
};

const getCachedGetIByJGetter = (tensor) => {
  const { shape, dims, offset } = tensor;

  if (tensor.isContiguous()) {
    if (offset === 0) {
      return () => () => (j) => j;
    } else {
      return (s0, o) => () => (j) => o + j;
    }
  } else if (dims === 1) {
    if (offset === 0) {
      return (s0) => () => (j) => s0 * j;
    } else {
      return (s0, o) => () => (j) => o + s0 * j;
    }
  }

  return (...nums) => (jStrides, jMods) => {
    //eliminate security concerns of `new Function`
    generalUtils.checkNumbers(nums);
    generalUtils.checkNumbers(jStrides);
    generalUtils.checkNumbers(jMods);

    const strides = nums.slice(0, nums.length - 1);
    const offset = nums[nums.length - 1];
    const terms = [];
    if (offset !== 0) {
      terms.push(offset);
    }

    for (let idx = 0; idx < dims; idx++) {
      const stride = strides[idx];
      if (stride !== 0 && shape[idx] !== undefined && shape[idx] > 1) {
        const moded = idx === 0 ?
          'j' :
          `(j%${jMods[idx]})`;
        const jStride = jStrides[idx];
        const floor = jStride === 1 ?
          moded :
          `Math.floor(${moded}/${jStride})`;
        if (stride === 1) {
          terms.push(floor);
        } else {
          terms.push(`${stride}*${floor}`);
        }
      }
    }

    if (terms.length === 0) {
      return () => 0;
    }

    const fnText = `return ${terms.join('+')}`;
    return new Function('j', fnText); //eslint-disable-line no-new-func
  };
};

const getCachedGetterData = (tensor) => {
  return {
    cachedGetI: getCachedGetI(tensor),
    cachedGetIByJGetter: getCachedGetIByJGetter(tensor)
  };
};

const buildTensorGettersFromCached = (cached, tensor) => {
  const nums = getTensorNumbers(tensor);
  return {
    getI: cached.cachedGetI(...nums),
    getIByJGetter: cached.cachedGetIByJGetter(...nums)
  };
};

module.exports = {
  buildTensorGetters(tensor) {
    const tensorCase = getTensorCase(tensor);

    if (!cache[tensorCase]) {
      cache[tensorCase] = getCachedGetterData(tensor);
    }

    return buildTensorGettersFromCached(cache[tensorCase], tensor);
  }
};
