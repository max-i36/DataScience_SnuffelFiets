import numpy as np


class ErrorUnpacker:
    @staticmethod
    def filter_out_error_codes(error_input_array, error_codes):
        if isinstance(error_input_array, list):
            error_input_array = np.array(error_input_array)

        error_code = sum(error_codes)

        filtered_indices = np.where(error_input_array & error_code == 0)

        return filtered_indices

    @staticmethod
    def get_error_array(error_code):
        if error_code == 0:
            return None
        else:
            error_array = []
            for i in range(0, 16):
                if error_code & 2**i != 0:
                    error_array.append(2**i)
            return error_array

    @staticmethod
    def print_error_info(error_code):
        error_descriptions = ['ACCELEROMETER ERROR 1: Sensor Not Found',
                              'Reserved',
                              'BME ERROR 1: Sensor Not Found',
                              'BME ERROR 2: Failed to begin reading',
                              'GPS ERROR 1: Sensor Not Found',
                              'GPS ERROR 2: No GPS Fix',
                              'Reserved',
                              'NO2 ERROR 1: Sensor Not Found',
                              'Reserved',
                              'PM ERROR 1: Sensor Not Found',
                              'PM ERROR 2a: Measurement Start Failure',
                              'PM ERROR 2b: Measurement Read Failure',
                              'PM ERROR 2c: Measurement Accuracy Uncertain',
                              'Reserved',
                              'Reserved',
                              'Reserved',
                              ]
        error_notes = ['Critical Error',
                       None,
                       'Critical Error',
                       'Critical Error',
                       'Critical Error',
                       'Allowed Error, device maybe indoors',
                       None,
                       'Should never be seen, NO2 removed in software.',
                       None,
                       'Critical Error',
                       'Critical Error',
                       'Allowed Error, PM sensors not ready. Wait 5 more measurements.',
                       'Critical Error',
                       None,
                       None,
                       None,
                       ]

        if error_code == 0:
            print('No error found.')
            return

        for i in range(0, 16):
            if error_code & 2**i != 0:
                if error_notes[i] is not None:
                    error_note = error_notes[i]
                else:
                    error_note = ''
                print("{:<12} {:<6} {:<45} {:<70}".format('Error Code', 2**i, error_descriptions[i], error_note))
        return

    @staticmethod
    def print_error_report(error_input_array):
        if isinstance(error_input_array, list):
            error_input_array = np.array(error_input_array)

        error_descriptions = ['ACCELEROMETER ERROR 1: Sensor Not Found',
                              'Reserved',
                              'BME ERROR 1: Sensor Not Found',
                              'BME ERROR 2: Failed to begin reading',
                              'GPS ERROR 1: Sensor Not Found',
                              'GPS ERROR 2: No GPS Fix',
                              'Reserved',
                              'NO2 ERROR 1: Sensor Not Found',
                              'Reserved',
                              'PM ERROR 1: Sensor Not Found',
                              'PM ERROR 2a: Measurement Start Failure',
                              'PM ERROR 2b: Measurement Read Failure',
                              'PM ERROR 2c: Measurement Accuracy Uncertain',
                              'Reserved',
                              'Reserved',
                              'Reserved',
                              ]

        error_notes = ['Critical Error',
                       None,
                       'Critical Error',
                       'Critical Error',
                       'Critical Error',
                       'Allowed Error, device maybe indoors',
                       None,
                       'Should never be seen, NO2 removed in software.',
                       None,
                       'Critical Error',
                       'Critical Error',
                       'Allowed Error, PM sensors not ready. Wait 5 more measurements.',
                       'Critical Error',
                       None,
                       None,
                       None,
                       ]

        error_counts = np.zeros(16)

        for error_code in error_input_array:
            for i in range(0, 16):
                if error_code & 2 ** i != 0:
                    error_counts[i] += 1

        for i in range(0, 16):
            if error_notes[i] is not None:
                error_note = error_notes[i]
            else:
                error_note = ''
            print("{:<10} {:<6} {:<6} {:<10} {:<45} {:<70}".format('Error Code', 2 ** i, 'Count:', int(error_counts[i]), error_descriptions[i], error_note))
        return
