import * as tf from '@tensorflow/tfjs';

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
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);

  output.append(canvas);
}

export async function bindPage() {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

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