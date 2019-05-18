module.exports = {
  timeAndReturn: (op, name) => {
    const t = new Date();
    const res = op();
    console.log(`${name} step took ${new Date() - t}ms.`);
    return res;
  },
  time: (op, name) => {
    module.exports.timeAndReturn(op, name);
  },
}
