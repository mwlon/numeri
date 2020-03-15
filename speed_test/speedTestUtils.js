module.exports = {
  timeAndReturn: (op, name, expected) => {
    const t = new Date();
    const res = op();
    const dt = new Date() - t;
    const ratio = dt / expected;

    let color;
    if (ratio < 1.2) {
      color = '\x1b[32m'; //green
    } else if (ratio < 1.5) {
      color = '\x1b[33m'; //yellow
    } else {
      color = '\x1b[31m'; //red
    }

    console.log( //eslint-disable-line no-console
      `${name} step ${color}[${new Date() - t} vs ${expected} ms]\x1b[0m`
    );
    return res;
  },

  time: (op, name, expected) => {
    module.exports.timeAndReturn(op, name, expected);
  },
};
