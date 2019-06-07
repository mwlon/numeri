module.exports = {
  prod(vals) {
    let res = 1;
    for (var i = 0; i < vals.length; i++) {
      const val = vals[i];
      res *= val === undefined ? 1 : val;
    }
    return res;
  },

  defaults(options, defaults) {
    if (!options) {
      return defaults;
    }

    return Object.assign({}, defaults, options);
  },
};
