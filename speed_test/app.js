const express = require('express');

const port = 3000;
const app = express();
//<script type="text/javascript" src="temp.js" />
app.get('/', (req, res) => {
  res.send(`
    <html>
      <head>
        <script type="text/javascript" src="main.js">
        </script>
      </head>
      <body>
        Check console
      </body>
    </html>
  `);
});
app.use(express.static('dist'));

app.listen(port, () => {
  console.log(`listening on ${port}`);
});
