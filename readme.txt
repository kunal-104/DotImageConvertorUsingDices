# Dice Image Generator Project

This project converts images into dice patterns using TensorFlow.js. It includes:
1. Data generation for training a model
2. Model training
3. Image conversion to dice representation

## Project Structure
```
dice-generator/
├── package.json
├── src/
│   ├── data/
│   │   └── dice.json
│   ├── generate-data.js
│   ├── train-model.js
│   └── dicify.js
└── web/
    ├── index.html
    ├── style.css
    └── app.js
```

## Setup Instructions

1. Create the project structure above
2. Install dependencies: `npm install`
3. Generate training data: `node src/generate-data.js`
4. Train the model: `node src/train-model.js`
5. Open web/index.html to use the dice image converter













# Navigate to your project directory
cd dice-generator

# Initialize the project and install dependencies 
node setup.js

# Generate the training data
npm run generate-data

# Train the model
npm run train

# Start the web server
npm start