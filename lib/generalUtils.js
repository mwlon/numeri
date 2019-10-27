module.exports = {
  prod(vals) {
    let res = 1;
    for (let i = 0; i < vals.length; i++) {
      const val = vals[i];
      res *= val === undefined ? 1 : val;
    }
    return res;
  },

  argmax(vals, by=(x) => x) {
    let arg = 0;
    let max = by(vals[0]);

    for (let i = 1; i < vals.length; i++) {
      const mapped = by(vals[i]);
      if (mapped > max) {
        max = mapped;
        arg = i;
      }
    }

    return {max, arg};
  },

  defaults(options, defaults) {
    if (!options) {
      return defaults;
    }

    return Object.assign({}, defaults, options);
  },

  checkNumber(x, msg) {
    if (typeof x !== 'number') {
      throw new Error(msg || `${x} is not a number!`);
    }
  },

  checkNumbers(xs, msg) {
    for (x of xs) {
      module.exports.checkNumber(x, msg);
    }
  }
};
