import tensorflow as tf


def write_tb_log(writer, obj_res, epoch):
    # Log each metric
    with writer.as_default():
        for key, value in obj_res.items():
            tf.summary.scalar(key, value, step=epoch)

    # Close the writer when done
    writer.close()
