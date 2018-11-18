# deepwater
Deep-learning based enhancer for underwater pictures

## High level APIs
* Generate a database based on standard format pictures: 
```
    python generate_database.py --database_source_path $TOPDIR/database/source --database_path $TOPDIR/database/tfrecords
```
* Train the network: 
```
    python deepwater.py --input_path $TOPDIR/input --database_path $TOPDIR/database/tfrecords --output_path $TOPDIR/output --mode train
```
* Process new images: 
```
    python deepwater.py --input_path $TOPDIR/input --database_path $TOPDIR/database/tfrecords --output_path $TOPDIR/output --mode predict
```


## Database
### Generate the training database
* Gather good quality images in a source folder `$TOPDIR/database/source`
* Split the source images into 3 subfolders `train`, `validation` and `test` under `$TOPDIR/database/source`. A good ratio is 80 / 20 / 20.
* Converge the source images into tfrecords (cf. `generate_database.py` API)
