const numeri = require('../../lib/index.js');
const { time } = require('../speedTestUtils');

const data = Array(100);
const n = 1000000;

time(() => {
  let x;
  for (let i = 0; i < n; i++) {
    x = numeri.Tensor.fromDataAndShape(data, []);
  }
}, 'from_data_and_shape_dim_0', 100);

time(() => {
  let x;
  for (let i = 0; i < n; i++) {
    x = numeri.Tensor.fromDataAndShape(data, [4]);
  }
}, 'from_data_and_shape_dim_1', 180); //not sure why slow sometimes

time(() => {
  let x;
  for (let i = 0; i < n; i++) {
    x = numeri.Tensor.fromDataAndShape(data, [4, 4]);
  }
}, 'from_data_and_shape_dim_2', 210);

time(() => {
  let x;
  for (let i = 0; i < n; i++) {
    x = new numeri.Tensor({
      data,
      shape: [4, 4],
      offset: 1,
      strides: [2, 8],
      mods: [8, 100]
    });
  }
}, 'complicated_constructor_dim_2', 240);

time(() => {
  let x;
  for (let i = 0; i < 1000000; i++) {
    x = numeri.Tensor.fromDataAndShape(data, [4, 4, 4]);
  }
}, 'from_data_and_shape_dim_3', 280);