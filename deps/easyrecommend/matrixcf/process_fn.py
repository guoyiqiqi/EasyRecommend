#! /usr/bin/env python
# -*-coding=utf8-*-


class InputFn:
    
    def __init__(self, local_ps):
        self.feature_len = 2
        self.label_len = 1
        self.n_parse_threads = 4
        self.shuffle_buffer_size = 1024
        self.prefetch_buffer_size = 1
        self.batch = 8
        self.local_ps = local_ps
        
    def input_fn(self, data_dir, is_test=False):
        def _parse_example(example):
            features = {
                "feature": tf.io.FixedLenFeature(self.feature_len, tf.int64),
                "label": tf.io.FixedLenFeature(self.label_len, tf.float32),
            }
            return tf.io.parse_single_example(example, features)
        
        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.compat.v1.py_func(self.local_ps.pull, [keys], tf.float32)
            result = {
                "feature": parsed["feature"],
                "label": parsed["label"]
            }
            return tf.io.parse_single_example(example, features)
        
        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.compat.v1.py_func(self.local_ps.pull, [keys], tf.float32)
            result = {
                "feature": parsed["feature"],
                "label": parsed["label"],
                "feature_embedding": keys_array,
            }
            return result
        
        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))
        
        dataset = tf.compat.v1.data.Dataset.list_files(files)
        # 数据复制多少份
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()
        # 读取tfrecord数据
        dataset = dataset.interleave(
            lambda _: tf.compat.v1.data.TFRecordDataset(_),
            cycle_length=1
        )
        # 对tfrecord的数据进行解析
        dataset = dataset.map(
            _parse_example,
            num_parallel_calls=self.n_parse_threads)
        
        # batch data
        dataset = dataset.batch(
            self.batch, drop_remainder=True)
        
        dataset = dataset.map(
            _get_embedding,
            num_parallel_calls=self.n_parse_threads)
        
        # 对数据进行打乱
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)
            
        # 数据预加载
        dataset = dataset.prefetch(
            buffer_size=self.prefetch_buffer_size)
        
        # 迭代器
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        return iterator, iterator.get_next() 