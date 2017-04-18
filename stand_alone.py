from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from argsource import args
import tensorflow as tf
import numpy as np
import argparse
import json
import base64


def main():
    with open('/home/jason/Documents/CMPS-4720-6720/Dataset/personal_images/DSC_4963_sized.jpg', "rb") as f:
        input_data = f.read()
    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args['stand_alone'] + "/export.meta")
        saver.restore(sess,args['stand_alone'] +"/exports")
        input = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
        output = tf.get_default_graph().get_tensor_by_name('packed:0')
        input_value = np.array(input_instance["input"])
        output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

    output_instance = dict(output=output_value.decode("ascii"), key="0")
    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

    with open(args['personal_test'] + "/out.jpg", "wb") as f:
        f.write(output_data)
        f.close()

main()





