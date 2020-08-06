class ShapeError extends Error {
  constructor(msg) {
    super(msg);
    this.name = this.constructor.name;
  }
}

module.exports = {
  ShapeError,
};
