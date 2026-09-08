
import datetime

class Log:
    def __init__(self, output_filename):
        self.output_filename = output_filename

    def __get_datetime_stamp(self):
        now = datetime.datetime.now()
        return now.strftime("%Y_%m_%d %H:%M:%S")

    def add_datetime(self, filename):
        formatted_date = self.__get_datetime_stamp()
        with open(filename, "a") as out_file:
            out_file.writelines([formatted_date + "\n"])

    def to_log(self, msg, display=False, add_time=True):
        if display:
            print(msg)
        if add_time:
            msg = self.__get_datetime_stamp() + "\t" + msg
        with open(self.output_filename, "a") as out_file:
            out_file.writelines([msg + "\n"])
