# Web viewer for deepwater

This is a sinple web viewer for the [deepwater project](https://github.com/annemenini/deepwater). Currently available at https://deepwater-project.web.app

It allows users to uplaod their own photos to be enhanced with the deepwater model running in the browser with TensorflowJS.

## Prerequisites

This web app depends on the trained model being publicly available at a given URL. To learn how to generate the web model, see the [main README](../README.md#converting-the-model-for-the-web).

For example, you can host the web model in a piblic [Google Cloud Storage](https://cloud.google.com/storage/) bucket. Copy your local files to the bucket with: 

    gsutil cp model/web/* gs://deepwater/model/v1

Update `config.js` with your own base URL and model version number.  

## Develop

Install dependencies with `npm i`

Start a local server with `npm run watch`

## Deploy

Build the production webapp with `npm run build`, then host the built file on a static file hosting service.

## Continuous deployment to Firebase hosting

This repo is configured to automatically be deployed to a [Firebase Hosting](https://firebase.google.com/docs/hosting/) static website on every change.

To do this, we use [Google Cloud Build](https://cloud.google.com/cloud-build/) to trigger a build every time a new change is pushed to the `web` folder of this repo.

You need to build a custom `firebase` builder, to do this, follow the instructions under the ["Firebase" tab here](https://cloud.google.com/cloud-build/docs/configuring-builds/build-test-deploy-artifacts#deploying_artifacts).

The actions to perform on deployment are specified in the `cloudbuild.yaml` file.

To continuously deploy to Firebase Hosting, we created a [Cloud Build trigger](https://cloud.google.com/cloud-build/docs/running-builds/automate-builds) that executes on every file change in the `web/` folder of this repository.
When creating this trigger, we set the [substitution value](https://cloud.google.com/cloud-build/docs/configuring-builds/substitute-variable-values) `_FIREBASE_TOKEN` with the token that we got when running `firebase login:ci`.
