# Enable once inference on one image and one target pose

Difference with the original one: main_hunter.py, trainer.py, trainer256.py, run_dftest_hunter.py, run_markettest_hunter.py, hunter_test.

The hunter_test folder contains the method and scripts for generating your own target pose heat maps.

The other four python files are adjusted to do once inference on your own image and your own target pose. 

Change the image path and target pose path in the main_hunter.py. 
Change the model direction in the .sh files for the checkpoint file.

Run the run_dftest_hunter.sh for Deep Fashion based generator, run_markettest_hunter.py for Market1501 based generator.
