const fs = require('fs');

describe('README.md', () => {
  it('contains only runnable code blocks', () => {
    const allContent = fs.readFileSync('README.md').toString();
    const codeBlocks = allContent
      .split('```')
      .filter((block, i) => i % 2 === 1);

    const codeContent = "const numeri = require('./lib')\n" +
      codeBlocks.join('\n');
    fs.writeFileSync('readmeCode.js', codeContent);
    require('../readmeCode');
  });
});
