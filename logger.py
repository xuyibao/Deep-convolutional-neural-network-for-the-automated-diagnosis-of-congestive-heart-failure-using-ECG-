import logging

class log_info():
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG,
                            filename='./deep_cnn/model/log.txt',
                            filemode='a',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def logger(self,info):
        logging.info(info)

