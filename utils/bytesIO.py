from io import BytesIO
import pickle


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 100000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples
