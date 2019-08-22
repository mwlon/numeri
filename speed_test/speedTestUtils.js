module.exports = {
  timeAndReturn: (op, name, expected) => {
    const t = new Date();
    const res = op();
    const dt = new Date() - t;
    const ratio = dt / expected;

    let color;
    if (ratio < 1.2) {
      color = '\x1b[32m';
    } else if (ratio < 1.5) {
      color = '\x1b[33m';
    } else {
      color = '\x1b[31m';
    }

    console.log(`${name} step ${color}[${new Date() - t} vs ${expected} ms]\x1b[0m`);
    return res;
  },
  time: (op, name, expected) => {
    module.exports.timeAndReturn(op, name, expected);
  },
}
