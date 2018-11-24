import * as tf from '@tensorflow/tfjs';

export async function bindPage() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main').style.display = 'block';
  }

bindPage();