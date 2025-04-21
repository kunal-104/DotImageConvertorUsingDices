const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const trainModel = async () => {
  // Load augmented data
  const dataPath = path.join(__dirname, 'data', 'dice_data.json');
  if (!fs.existsSync(dataPath)) {
    console.error('No training data found. Run generate-data.js first.');
    process.exit(1);
  }

  const diceData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  const numOptions = Object.keys(diceData).length;
  
  // Stack and shuffle data
  let diceImages = [].concat(
    ...Object.keys(diceData).map(key => diceData[key])
  );
  
  // Create labels
  let answers = [];
  Object.keys(diceData).forEach(key => {
    const valueArray = new Array(diceData[key].length).fill(parseInt(key));
    answers.push(...valueArray);
  });

  const shuffleArrays = (array1, array2) => {
    if (array1.length !== array2.length) {
      throw new Error('Arrays must be of equal length to shuffle');
    }
    
    const length = array1.length;
    
    for (let i = length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      
      // Swap elements in array1
      [array1[i], array1[j]] = [array1[j], array1[i]];
      
      // Swap elements in array2
      [array2[i], array2[j]] = [array2[j], array2[i]];
    }
    
    return [array1, array2];
  };
  
  // Shuffle the data and answers together
  [diceImages, answers] = shuffleArrays(diceImages, answers);
  
  // Split into training and testing data
  const splitIdx = Math.floor(diceImages.length * 0.8);
  const trainImages = diceImages.slice(0, splitIdx);
  const testImages = diceImages.slice(splitIdx);
  const trainAnswers = answers.slice(0, splitIdx);
  const testAnswers = answers.slice(splitIdx);
  
  // Convert to tensors
  const trainX = tf.tensor(trainImages).expandDims(3);
  const testX = tf.tensor(testImages).expandDims(3);
  const trainY = tf.oneHot(trainAnswers, numOptions);
  const testY = tf.oneHot(testAnswers, numOptions);
  
  // Create model
  const model = tf.sequential();
  const inputShape = [12, 12, 1];
  
  model.add(tf.layers.flatten({ inputShape }));
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({
    units: numOptions,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax',
  }));
  
  // Compile the model
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  // Train the model
  console.log('Training model...');
  await model.fit(trainX, trainY, {
    epochs: 10,
    batchSize: 32,
    validationData: [testX, testY],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} - loss: ${logs.loss.toFixed(4)}, accuracy: ${logs.acc.toFixed(4)}`);
      }
    }
  });
  
  // Evaluate model
  const evalOutput = model.evaluate(testX, testY);
  console.log(`Evaluation loss: ${evalOutput[0].dataSync()[0].toFixed(4)}`);
  console.log(`Evaluation accuracy: ${evalOutput[1].dataSync()[0].toFixed(4)}`);
  
  // Save the model
  const modelDir = path.join(__dirname, '..', 'web', 'model');
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }
  
  await model.save(`file://${modelDir}`);
  console.log(`Model saved to ${modelDir}`);
  
  // Save the original dice data for the web app
  const webDataDir = path.join(__dirname, '..', 'web', 'data');
  if (!fs.existsSync(webDataDir)) {
    fs.mkdirSync(webDataDir, { recursive: true });
  }
  
  const originalDice = require(path.join(__dirname, 'data', 'dice.json')).data;
  fs.writeFileSync(
    path.join(webDataDir, 'dice.json'), 
    JSON.stringify({ data: originalDice })
  );
  console.log('Original dice data saved for web app');
  
  // Clean up
  tf.dispose([trainX, testX, trainY, testY]);
};

try {
  trainModel();
} catch (e) {
  console.error('ERROR:', e);
}