import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

import {modelURL, weightsURL} from './config.js'

const INPUT_NODE_NAME = 'degraded';
const OUTPUT_NODE_NAME = 'final_output';

const MAX_IMAGE_SIZE = 1024;

let model;

async function loadModel() {
  if(!model) {
    const loadingEl = document.getElementById('loading'); 
    loadingEl.style.display = 'block';
    console.log('model: loading');
    model = await loadFrozenModel(modelURL, weightsURL);
    console.log('model: loaded');
    loadingEl.style.display = 'none';
  }
};

function handleFiles(files) {
  for (let i = 0; i < files.length; i++) {
    let file = files[i];
    
    if (!file.type.startsWith('image/')){ continue }
    
    let reader = new FileReader();
    reader.onload =  e => { processFile(e.target.result); };
    reader.readAsDataURL(file);
  }
}

function processFile(url) {
  let img = document.createElement("img");
  img.onload = function() {
    processImg(this);
  }
  img.src = url;
}


function getNewCanvasFromImage(img) {
  let canvas = document.createElement("canvas");

  if(img.naturalWidth > MAX_IMAGE_SIZE) {
    canvas.width = MAX_IMAGE_SIZE;
    canvas.height = img.naturalHeight * MAX_IMAGE_SIZE / img.naturalWidth;
  } else if(img.naturalHeight > MAX_IMAGE_SIZE) {
    canvas.width = img.naturalWidth * MAX_IMAGE_SIZE / img.naturalHeight;
    canvas.height = MAX_IMAGE_SIZE;
  } else {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
  }

  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);

  return canvas;
}

function getNewCanvasFromSize(canvas) {
  let newCanvas = document.createElement("canvas");
  newCanvas.width = canvas.width;
  newCanvas.height = canvas.height;
  return newCanvas;
}

function appendCanvas(canvas) {
  let output = document.getElementById("output");
  output.append(canvas);
}

function processImg(img) {
  // we copy into a canvas because https://github.com/tensorflow/tfjs/issues/1111
  let canvasInput = getNewCanvasFromImage(img);
  canvasInput.className = "input-preview";
  appendCanvas(canvasInput);

  const demoFloat32 = tf.fromPixels(canvasInput).asType('float32');
  console.log('model: executing');
  let result = model.execute({[INPUT_NODE_NAME]: demoFloat32}, OUTPUT_NODE_NAME);
  console.log('model: executed');

  let canvasResult = getNewCanvasFromSize(canvasInput)
  canvasResult.className = "result-preview";

  tf.toPixels(result.asType('int32').squeeze([0]), canvasResult);

  appendCanvas(canvasResult);
}

export async function bindPage() {
  document.querySelectorAll('.demo').forEach((img) => {
    img.addEventListener("click", async () => {
      await loadModel();
      processImg(img);
    }, false);
  });

  const inputElement = document.getElementById("input");
  inputElement.addEventListener("change", async (event) => {
    await loadModel();
    handleFiles(event.target.files);
  }, false);

  const dropzone = document.getElementById("dropbox");

  dropzone.ondragover = dropzone.ondragenter = (event) => {
    event.stopPropagation();
    event.preventDefault();
  }
  dropzone.ondrop = async (e) => {
    e.stopPropagation();
    e.preventDefault();
    var dt = e.dataTransfer;
    var files = dt.files;
  
    await loadModel();
    handleFiles(files);
  }
}

bindPage();