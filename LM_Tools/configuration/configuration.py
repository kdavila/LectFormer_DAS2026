import ast
import json
import warnings

class Configuration:
    SubTypeAny = 0
    SubTypeString = 1
    SubTypeInt = 2
    SubTypeFloat = 3
    SubTypeBool = 4

    def __init__(self, config_data, strict_mode=False, warning_mode=True):
        self.data = config_data
        self.strict_mode = strict_mode
        self.warning_mode = warning_mode

    def get(self, name, default=None):
        return self.__get(name, Configuration.SubTypeAny, default)

    def __get(self, name, subtype, default=None):
        found, raw_value = self.__find_key(name)

        if found:
            # key found ...
            if subtype == Configuration.SubTypeAny:
                # let python guess the right class
                try:
                    value = ast.literal_eval(raw_value)
                except:
                    # interpret as string
                    value = raw_value

                if isinstance(value, dict):
                    warnings.warn(
                        f"Configuration key '{name}' contains a dictionary. "
                        f"Consider using get_subconfig() instead.",
                        stacklevel=2
                    )

                return value
            elif subtype == Configuration.SubTypeString:
                return raw_value
            elif subtype == Configuration.SubTypeInt:
                return int(raw_value)
            elif subtype == Configuration.SubTypeFloat:
                return float(raw_value)
            elif subtype == Configuration.SubTypeBool:
                return int(raw_value) > 0
            else:
                raise Exception("Unknow Configuration Data Type")
        else:
            # key not found, behavior depends on current settings
            if self.strict_mode:
                # throw an error
                raise Exception(f"Key {name} not found in configuration")
            else:
                if self.warning_mode:
                    print(f"- WARNING: Key {name} not found in configuration, using default value: {default}")
                return default


    def __find_key(self, full_key):
        # assumes that keys will be given as "a.b.c", where this represents
        # {"a":{ "b": {"c": ...}}}
        parts = full_key.split(".")
        # find the subkey
        current_dict = self.data
        for part in parts:
            if part in current_dict:
                current_dict = current_dict[part]
            else:
                # failed to find the required value ...
                return False, None

        return True, current_dict

    def get_str(self, name, default=""):
        return self.__get(name, Configuration.SubTypeString, default)

    def get_int(self, name, default=0):
        return self.__get(name, Configuration.SubTypeInt, default)

    def get_float(self, name, default=0.0):
        return self.__get(name, Configuration.SubTypeFloat, default)

    def get_bool(self, name, default=False):
        # This assumes string is integer 1 or 0 (True, False)
        return self.__get(name, Configuration.SubTypeBool, default)

    def get_names(self):
        return self.data.keys()

    def set(self, name, value, add_missing_keys=False):
        parts = name.split(".")
        # find the subkey
        current_dict = self.data
        for part in parts[:-1]:
            if part in current_dict:
                current_dict = current_dict[part]
            else:
                if not add_missing_keys:
                    # report the missing key as an error
                    raise Exception(f"Could not find sub-key {part} in {name}")
                else:
                    # add the missing key
                    current_dict[part] = {}
                    # and then move ...
                    current_dict = current_dict[part]

        current_dict[parts[-1]] = value

        print(self.data)

    def contains(self, name):
        found, _ = self.__find_key(name)
        if not found and self.warning_mode:
            print(f"- Warning: Searched for {name} in Configuration, but it was not found!")
        return found

    def save(self, filename):
        # save current configuration status to output file ...
        with open(filename, "w", encoding="utf8") as out_file:
            json.dump(self.data, out_file, indent=4)

    def get_subconfig(self, name):
        parts = name.split(".")
        # find the subkey
        current_dict = self.data
        for part in parts:
            if part in current_dict:
                current_dict = current_dict[part]
            else:
                # sub-config not found ...
                return None

        return Configuration(current_dict, self.strict_mode, self.warning_mode)

    @staticmethod
    def from_file(filename, strict_mode=False, warning_mode=True):
        with open(filename, "r") as in_file:
            config_data = json.load(in_file)

        return Configuration(config_data, strict_mode, warning_mode)


