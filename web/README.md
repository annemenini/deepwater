# Web viewer for deepwater

This is a sinple web viewer for the [deepwater project](https://github.com/annemenini/deepwater).

It allows users to uplaod their own photos to be enhanced with the deepwater model running in the browser with TensorflowJS.

## Prerequisites

This web app depends on the trained model being publicly available at a given URL. To learn how to generate the web model, see the [main README](../README.ms#converting-the-model-for-the-web).

For example, you can host the web model in a piublic [Google Cloud Storage](https://cloud.google.com/storage/) bucket.

Update `config.js` with your own base URL and model version number.  

## Develop

Install dependencies with `npm i`

Start a local server with `npm run watch`

## Deploy

Build the production webapp with `npm run build`