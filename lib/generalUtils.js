module.exports = {
  prod(vals) {
    let res = 1;
    for (var i = 0; i < vals.length; i++) {
      res *= vals[i] || 1;
    }
    return res;
  }
};
