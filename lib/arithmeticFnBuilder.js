//ok this takes some explanation
//I need to dynamically produce the getter functions for tensors
//because
//  return loc0 + stride1 * loc1;
//is much better than
//  let s = 0; for (let idx = 0; idx < dims; idx++) {s += loc[idx] * stride[idx]}; return s;
//This object helps create efficient functions on the fly, and avoid creating
//similar functions twice, since each one can require milliseconds to format
//and eval.

const cache = {};
let totalT = 0;

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

const getTensorFnTexts = (tensor) => {
  const { strides, offset, shape, dims } = tensor;

  const fnArgs = Array(dims);
  const getITerms = [];
  const getIByJTerms = [];
  const numArgs = Array(dims + 1);
  const jStrideArgs = Array(dims);
  const jModArgs = Array(dims);
  numArgs[dims] = 'o';
  if (offset !== 0) {
    getITerms.push('o');
    getIByJTerms.push('o');
  }

  for (let idx = 0; idx < dims; idx++) {
    const argName = `l${idx}`;
    fnArgs[idx] = argName;
    const numArg = `n${idx}`;
    numArgs[idx] = numArg;
    const jStride = `s${idx}`;
    jStrideArgs[idx] = jStride;
    const jMod = `m${idx}`;
    jModArgs[idx] = jMod;

    const stride = strides[idx];
    if (stride !== 0 && shape[idx] !== undefined && shape[idx] > 1) {
      const floor = `Math.floor((j%${jMod})/${jStride})`;
      if (stride === 1) {
        getITerms.push(argName);
        getIByJTerms.push(floor);
      } else {
        getITerms.push(`${numArg}*${argName}`);
        getIByJTerms.push(`${numArg}*${floor}`);
      }
    }
  }

  if (getITerms.length === 0) {
    return {
      getI: '()=>()=>0',
      getIByJGetter: '()=>()=>()=>0'
    };
  }

  const getI = `(${numArgs.join(',')})=>(${fnArgs.join(',')})=>` +
    `${getITerms.join('+')}`;

  let getIByJGetter;
  if (tensor.isContiguous()) {
    if (offset === 0) {
      getIByJGetter = '()=>()=>j=>j';
    } else {
      getIByJGetter = '(s0,o)=>()=>j=>o+j';
    }
  } else if (dims === 1) {
    if (offset === 0) {
      getIByJGetter = 's0=>()=>j=>s0*j';
    } else {
      getIByJGetter = '(s0,o)=>()=>j=>o+s0*j';
    }
  } else {
    getIByJGetter = `(${numArgs.join(',')})=>(ss,ms)=>{` +
    `const [${jStrideArgs.join(',')}]=ss;` +
    `const [${jModArgs.join(',')}]=ms;` +
    `return j=>${getIByJTerms.join('+')};` +
    '}';
  }
  console.log(getIByJGetter)

  return {getI, getIByJGetter};
};

module.exports = {
  buildTensorGetters(tensor) {
    const tensorCase = getTensorCase(tensor);

    if (!cache[tensorCase]) {
      const fns = getTensorFnTexts(tensor);
      for (const key of Object.keys(fns)) {
        fns[key] = eval(fns[key]); //eslint-disable-line no-eval
      }
      cache[tensorCase] = fns;
    }

    const fns = cache[tensorCase];
    const nums = getTensorNumbers(tensor);
    const res = {};
    for (const key of Object.keys(fns)) {
      res[key] = fns[key](...nums);
    }
    return res;
  }
};
