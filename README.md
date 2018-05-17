# Simple object recognition using keras

## How to

Place training images in dataset/train/[label]/ and validation images in dataset/validate/[label]/ .
Where [label] is the name of the object. Add as many different [label]s as you want.

Train the network: 
`python train.py`

Visualize the model:
`python visualize.py`

Use the model on a single image:
`python file.py /path/to/file.jpg`

Run a simple web interface and api (on port 5353) that accept posted images:
`python web.py`
