import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

import {modelURL, weightsURL} from './config.js'

const INPUT_NODE_NAME = 'degraded';
const OUTPUT_NODE_NAME = 'final_output';

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

function drawImage(img) {
  console.log("drawing input");
  let output = document.getElementById("output");

  let canvas = document.createElement("canvas");
  canvas.className = "input-preview";
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);

  output.append(canvas);
}

function drawResult(originalImage, result) {
  console.log("drawing result");
  const output = document.getElementById("output");
  const canvas = document.createElement("canvas");
  canvas.className = "result-preview";
  canvas.width = originalImage.naturalWidth;
  canvas.height = originalImage.naturalHeight;

  tf.toPixels(result.asType('int32').squeeze(), canvas);

  output.append(canvas);
}

function processImg(img) {
  drawImage(img);

  const demoFloat32 = tf.fromPixels(img).asType('float32');
  console.log('model: executing');
  let result = model.execute({[INPUT_NODE_NAME]: demoFloat32}, OUTPUT_NODE_NAME);
  console.log('model: executed');

  drawResult(img, result);
}

export async function bindPage() {
  const loadButton = document.getElementById("run-example");
  loadButton.addEventListener("click", async () => {
    await loadModel();
    processImg(document.getElementById('demo'));
  }, false);

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