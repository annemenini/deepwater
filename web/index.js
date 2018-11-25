import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'https://storage.googleapis.com/deepwater/model/v1/tensorflowjs_model.pb';
const WEIGHTS_URL = 'https://storage.googleapis.com/deepwater/model/v1/weights_manifest.json';

const INPUT_NODE_NAME = 'degraded';
const OUTPUT_NODE_NAME = 'final_output';

function handleFiles(files) {
  for (let i = 0; i < files.length; i++) {
    let file = files[i];
    
    if (!file.type.startsWith('image/')){ continue }
    
    let reader = new FileReader();
    reader.onload =  e => { processImage(e.target.result); };
    reader.readAsDataURL(file);
  }
}

function processImage(url) {
  let img = document.createElement("img");
  img.onload = function() {
    drawImage(this);
  }
  img.src = url;
}

function drawImage(img) {
  let output = document.getElementById("output");

  let canvas = document.createElement("canvas");
  canvas.className = "input-preview";
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);

  output.append(canvas);
}

async function loadModel() { 
  console.log('model: loading');
  const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  console.log('model: loaded');

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  // TODO apply to all input
  //const input = document.querySelector(".input-preview");
  const demo = document.getElementById('demo');
  const demoFloat32 = tf.fromPixels(demo).asType('float32');
  console.log('model: executing');
  let result = model.execute({[INPUT_NODE_NAME]: demoFloat32}, OUTPUT_NODE_NAME);
  // result is a 4
  console.log('model: executed');

  let output = document.getElementById("demos");
  let canvas = document.createElement("canvas");
  canvas.width = demo.naturalWidth;
  canvas.height = demo.naturalHeight;

  tf.toPixels(result.asType('int32').squeeze(), canvas);
  output.append(canvas);

  console.log("now what?");
}

export async function bindPage() {
  let inputElement = document.getElementById("input");
  inputElement.addEventListener("change", function() {handleFiles(this.files)}, false);

  let dropzone = document.getElementById("dropbox");

  dropzone.ondragover = dropzone.ondragenter = function(event) {
    event.stopPropagation();
    event.preventDefault();
  }
  dropzone.ondrop = function drop(e) {
    e.stopPropagation();
    e.preventDefault();
    var dt = e.dataTransfer;
    var files = dt.files;
  
    handleFiles(files);
  }
}

bindPage();
loadModel();