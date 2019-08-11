module.exports = {
  prod(vals) {
    let res = 1;
    for (var i = 0; i < vals.length; i++) {
      const val = vals[i];
      res *= val === undefined ? 1 : val;
    }
    return res;
  },

  argmax(vals, by=(x) => x) {
    let arg = 0;
    let max = by(vals[0]);

    for (var i = 1; i < vals.length; i++) {
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
};
