const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Create required directories
const dirs = [
  'src/data',
  'web/data',
  'web/model'
];

dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
});

// Install dependencies
console.log('Installing dependencies...');
exec('npm install', (error, stdout, stderr) => {
  if (error) {
    console.error(`Error installing dependencies: ${error}`);
    return;
  }
  
  console.log(stdout);
  console.log('Dependencies installed successfully');
  
  // Run steps in sequence
  console.log('\nProject setup complete. Now you can run:');
  console.log('1. npm run generate-data  - Generate training data');
  console.log('2. npm run train         - Train the model');
  console.log('3. npm start             - Start the web server');
  console.log('\nAfter starting the server, open http://localhost:8080 in your browser');
});