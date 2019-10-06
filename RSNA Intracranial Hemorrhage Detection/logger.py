import logging

class Logger:
    __instance = None
    @staticmethod
    def get_instance():
        """ Static access method. """
        if Logger.__instance == None:
            Logger()
        return Logger.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Logger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
            logging.basicConfig(filename='loggOutput.log',level=logging.DEBUG)

    def log_error(self, message):
        print(message)
        logging.error(message)