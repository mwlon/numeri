const fs = require('fs');

describe('README.md', () => {
  it('contains only runnable code blocks', () => {
    const allContent = fs.readFileSync('README.md').toString();
    const codeBlocks = [];
    allContent.split('```').forEach((block, i) => {
      if (i % 2 === 1) {
        codeBlocks.push(block);
      }
    });

    const codeContent = "const numeri = require('./lib');\n" +
      codeBlocks.join('\n');
    fs.writeFileSync('readmeCode.js', codeContent);
    require('../readmeCode');
  });
});
