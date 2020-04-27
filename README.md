# Inferencing with Tensorflow model using TensorRT

The objective of this repo is to make it easy to go from trained model in TensorFlow in Python to inferencing in c++. There is a whole documentation linked in the [Bibliography Section](#bibliography) below. This is only an excerpt from there and a mix with other materials consulted. All referenced materials have been linked accordingly.

## TensorRT Download and installation

Follow the [link to download and install tensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#downloading).


## Steps (UFF Route)

1. Freeze Keras Model: Here is a sample code to freeze keras model.

```python
from keras.models import load_model
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

def convert_keras_to_pb(keras_model, out_names, models_dir, model_filename):
	model = load_model(keras_model)
	K.set_learning_phase(0)
	sess = K.get_session()
	saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
	checkpoint_path = saver.save(sess, 'saved_ckpt', global_step=0, latest_filename='checkpoint_state')
	graph_io.write_graph(sess.graph, '.', 'tmp.pb')
	freeze_graph.freeze_graph('./tmp.pb', '',
                          	False, checkpoint_path, out_names,
                          	"save/restore_all", "save/Const:0",
                          	models_dir+model_filename, False, "")
```

2. Convert frozen graph to UFF

```bash
convert-to-uff input_file [-o output_file] [-O output_node]

# You can list the TensorFlow layers:
convert-to-uff input_file -l

```

3. 

## Bibliography

1. NVDIA Documentation, Working with Tensorflow - https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_tf