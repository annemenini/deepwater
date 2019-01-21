# deepwater
Deep-learning based enhancer for underwater pictures

## High level APIs
* Generate a database based on standard format pictures: 
```
    python generate_database.py \
    --database_source_path $TOPDIR/database/source \
    --database_path $TOPDIR/database/tfrecords
```
* Train the network: 
```
    python deepwater.py \
    --input_path $TOPDIR/input \
    --database_path $TOPDIR/database/tfrecords \
    --output_path $TOPDIR/output --mode train
```
* Process new images: 
```
    python deepwater.py \
    --input_path $TOPDIR/input \
    --database_path $TOPDIR/database/tfrecords \
    --output_path $TOPDIR/output \
    --mode predict
```


## Database
### Generate the training database
* Gather good quality images in a source folder `$TOPDIR/database/source`
* Split the source images into 3 subfolders `train`, `validation` and `test` under `$TOPDIR/database/source`. A good ratio is 80 / 20 / 20.
* Converge the source images into tfrecords (cf. `generate_database.py` API)

## Converting the model for the web

Train the model, then copy the `.pb` and `variables` folder containing your model to `$TOPDIR/model/savedmodel`.

Then run the following to convert to a format that can be loaded by TensorFlowJS:

    tensorflowjs_converter \
        --input_format=tf_saved_model \
        --output_node_names='final_output' \
        $TOPDIR/model/savedmodel \
        $TOPDIR/model/web

The output files is what you need to serve publicly to use the web viewer. See [`README` file in the `web` folder](web/README.md) to read more about the web viewer.

## Configuration

There are 2 ways to tune the default coniguration: via command line options and via .json config file.

For each option, if it is not tuned by command line nor by the json config file, then the default value defined in configuration.py will be used. 

Therefore the json config file is optional.

If an option is tuned via both program option and json, the value defined in the json configuration file will be used.

An example of json file is provided in configuration_template.json.

To pass a configuration via json file, use the command line option `--config_file my_config.json`. Note that the config is expected to be found in the input folder which can be specified by the command line option `--input_path /my/input/path`.

## Training with GCP ML Engine
### Local training (from the root of the repo):
```
gcloud ml-engine local train \
--module-name trainer.deepwater \
--package-path trainer/ \
--job-dir ../../output_gcloud \
-- \
--input_path ../../input \
--output_path ../../output
```
### Cloud training (from the root of the repo):
Create bucket:

    gsutil mb -l us-central1 gs://deepwater-project-mlengine

Upload tfrecords:

    gsutil cp -r database/tfrecords gs://deepwater-project-mlengine/data

Run training:
```
gcloud ml-engine jobs submit training deepwater_single_4 \
--job-dir gs://deepwater-project-mlengine/deepwater_single_4 \
--runtime-version 1.12 \
--module-name trainer.deepwater \
--package-path trainer/ \
--region us-central1 \
--scale-tier BASIC_GPU \
-- \
--database_path gs://deepwater-project-mlengine/data \
--output_path gs://deepwater-project-mlengine/deepwater_single_4
```
### Deploy model for prediction
Create model:
```
gcloud ml-engine models create deepwater_64epochs --regions=us-central1
```
Create a version:
```
gcloud ml-engine versions create v1 \
--model deepwater_64epochs \
--origin gs://deepwater-project-mlengine/deepwater_single_7/saved_model/1547444780/ \
--runtime-version 1.12
```
