
// Global variables
let model;
let originalDice;
let statusMessage = document.getElementById('statusMessage');
const numDiceInput = document.getElementById('numDice');
const processBtn = document.getElementById('processBtn');
const imageUpload = document.getElementById('imageUpload');
const originalCanvas = document.getElementById('originalCanvas');
const diceCanvas = document.getElementById('diceCanvas');
const originalCtx = originalCanvas.getContext('2d');
const diceCtx = diceCanvas.getContext('2d');

// Load model and original dice data when page loads
async function initialize() {
  updateStatus('Loading model and dice data...');
  
  try {
    // Load the model
    model = await tf.loadLayersModel('model/model.json');
    updateStatus('Model loaded successfully');
    
    // Load original dice data
    const response = await fetch('data/dice.json');
    const data = await response.json();
    originalDice = data.data.map(die => tf.tensor(die));
    
    updateStatus('Ready to generate dice images');
    processBtn.disabled = false;
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('Error loading model or dice data. See console for details.');
  }
}

// Update the status message
function updateStatus(message) {
  statusMessage.textContent = message;
  console.log(message);
}

// Process the selected image
async function processImage() {
  if (!imageUpload.files || imageUpload.files.length === 0) {
    updateStatus('Please select an image first');
    return;
  }
  
  const file = imageUpload.files[0];
  const numDice = parseInt(numDiceInput.value);
  
  if (numDice < 4 || numDice > 64) {
    updateStatus('Number of dice must be between 4 and 64');
    return;
  }
  
  updateStatus('Processing image...');
  processBtn.disabled = true;
  
  try {
    // Load the selected image
    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    img.onload = async () => {
      try {
        // Draw original image on canvas
        const size = Math.min(500, window.innerWidth - 40);
        originalCanvas.width = size;
        originalCanvas.height = size;
        
        // Calculate aspect ratio to maintain when drawing
        const scale = Math.min(size / img.width, size / img.height);
        const width = img.width * scale;
        const height = img.height * scale;
        const offsetX = (size - width) / 2;
        const offsetY = (size - height) / 2;
        
        originalCtx.fillStyle = 'white';
        originalCtx.fillRect(0, 0, size, size);
        originalCtx.drawImage(img, offsetX, offsetY, width, height);
        
        // Convert to tensor and process
        await tf.tidy(() => {
          // Create a tensor from the canvas
          const imgTensor = tf.browser.fromPixels(originalCanvas, 1)
            .div(255)  // Normalize to 0-1
            .greater(0.5) // Convert to binary (threshold at 0.5)
            .cast('float32');
          
          // Resize to match number of dice requested
          const preSize = numDice * 12; // Each die is 12x12
          const resized = tf.image.resizeNearestNeighbor(imgTensor, [preSize, preSize]);
          
          // Cut image into grid of patches
          const grid = [];
          for (let y = 0; y < numDice; y++) {
            for (let x = 0; x < numDice; x++) {
              const patch = resized.slice(
                [y * 12, x * 12, 0], 
                [12, 12, 1]
              ).reshape([12, 12]);
              
              grid.push(patch);
            }
          }
          
          // Stack all patches for batch prediction
          const stackedPatches = tf.stack(grid).expandDims(3);
          
          // Run prediction
          const predictions = model.predict(stackedPatches);
          const { indices } = tf.topk(predictions);
          const diceIndices = indices.dataSync();
          
          // Create output canvas
          diceCanvas.width = numDice * 12;
          diceCanvas.height = numDice * 12;
          
          // Reconstruct image with dice
          let currentDie = 0;
          for (let y = 0; y < numDice; y++) {
            for (let x = 0; x < numDice; x++) {
              // Get the predicted die
              const dieIndex = diceIndices[currentDie++];
              const dieTensor = originalDice[dieIndex];
              
              // Convert to pixel data
              const dieData = dieTensor.dataSync();
              
              // Draw the die on the canvas
              for (let dy = 0; dy < 12; dy++) {
                for (let dx = 0; dx < 12; dx++) {
                  const value = dieData[dy * 12 + dx];
                  diceCtx.fillStyle = value > 0.5 ? 'white' : 'black';
                  diceCtx.fillRect(x * 12 + dx, y * 12 + dy, 1, 1);
                }
              }
            }
          }
        });
        
        updateStatus('Dice image generated successfully');
      } catch (error) {
        console.error('Processing error:', error);
        updateStatus('Error processing image. See console for details.');
      } finally {
        processBtn.disabled = false;
      }
    };
    
    img.onerror = () => {
      updateStatus('Error loading the selected image');
      processBtn.disabled = false;
    };
  } catch (error) {
    console.error('Processing error:', error);
    updateStatus('Error processing image. See console for details.');
    processBtn.disabled = false;
  }
}

// Add event listeners
window.addEventListener('load', initialize);
processBtn.addEventListener('click', processImage);